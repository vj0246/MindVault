-- MindVault initial schema.
-- Reconstructed from the live Supabase project's actual current state
-- (introspected via information_schema/pg_catalog), not hand-written from
-- memory -- this is what's really running in production. Apply via the
-- Supabase SQL editor or `supabase db push` against a fresh project.
--
-- Note: only two small pieces of this (the documents.folder column and
-- the increment_token_usage RPC) were ever tracked in Supabase's own
-- migration history -- the rest of this schema was built directly against
-- the live project across earlier sessions and is being committed to the
-- repo, consolidated into one file, for the first time here.

create extension if not exists vector;

-- ── documents ────────────────────────────────────────────────────────────
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  filename text not null,
  chunk_count integer,
  uploaded_at timestamptz default now(),
  user_id uuid,
  folder text
);

-- ── chunks ───────────────────────────────────────────────────────────────
-- embedding is vector(384) -- matches fastembed's BAAI/bge-small-en-v1.5
-- output dimension. content_tsv is a generated column powering the
-- keyword-search half of the hybrid retrieval pipeline.
create table if not exists chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid references documents(id),
  content text not null,
  embedding vector(384),
  chunk_index integer,
  filename text,
  user_id uuid,
  page_number integer default 0,
  content_tsv tsvector generated always as (to_tsvector('english', content)) stored
);

create index if not exists chunks_content_tsv_idx on chunks using gin (content_tsv);
-- No vector index (HNSW/IVFFlat) on `embedding` yet -- see README's
-- "What I have not done" section. Fine at current data volume; will become
-- the biggest latency bottleneck once similarity search risks a
-- sequential scan at larger chunk counts.

-- ── chat messages (raw turn-by-turn history) ────────────────────────────
create table if not exists sessions (
  id uuid primary key default gen_random_uuid(),
  session_id text not null,
  role text not null,
  content text not null,
  timestamp timestamptz default now(),
  user_id uuid
);

-- ── chat sessions (thread metadata: naming, sharing) ────────────────────
create table if not exists chat_sessions (
  id text primary key,
  user_id uuid not null,
  name text default 'New Chat',
  created_at timestamptz default now(),
  last_active timestamptz default now(),
  share_token text unique,
  is_public boolean default false
);

-- ── knowledge graph ──────────────────────────────────────────────────────
create table if not exists graph_nodes (
  id uuid primary key default gen_random_uuid(),
  node_id text not null unique,
  sources text[],
  user_id uuid
);

create table if not exists graph_edges (
  id uuid primary key default gen_random_uuid(),
  source text not null,
  target text not null,
  relation text,
  user_id uuid
);

-- ── query result cache ───────────────────────────────────────────────────
-- Backend-only (service-role key bypasses RLS) -- no client ever reads
-- this table directly, so it has RLS enabled with intentionally zero
-- policies, which denies all non-service-role access by default.
create table if not exists query_cache (
  cache_key text primary key,
  result jsonb not null,
  created_at timestamptz default now(),
  expires_at timestamptz not null
);

-- ── rate limiting + token usage (shared counter table) ──────────────────
-- Same backend-only-access reasoning as query_cache.
create table if not exists rate_limits (
  key text primary key,
  count integer not null default 0,
  window_start timestamptz not null default now()
);

-- ── user preferences + long-term memory ─────────────────────────────────
create table if not exists user_preferences (
  user_id uuid primary key references auth.users(id),
  name text not null default '',
  tone text not null default 'Neutral',
  priorities text not null default '',
  system_prompt text not null default '',
  theme text not null default 'Light',
  updated_at timestamptz not null default now()
);

create table if not exists user_memory_notes (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id),
  content text not null,
  created_at timestamptz not null default now()
);

-- ── Row Level Security ───────────────────────────────────────────────────
alter table documents enable row level security;
alter table chunks enable row level security;
alter table sessions enable row level security;
alter table chat_sessions enable row level security;
alter table graph_nodes enable row level security;
alter table graph_edges enable row level security;
alter table query_cache enable row level security;
alter table rate_limits enable row level security;
alter table user_preferences enable row level security;
alter table user_memory_notes enable row level security;

create policy "users see own documents" on documents for all using (auth.uid() = user_id);
create policy "users see own chunks" on chunks for all using (auth.uid() = user_id);
create policy "users see own sessions" on sessions for all using (auth.uid() = user_id);
create policy "users own chat sessions" on chat_sessions for all using (auth.uid() = user_id);
create policy "users see own graph nodes" on graph_nodes for all using (auth.uid() = user_id);
create policy "users see own graph edges" on graph_edges for all using (auth.uid() = user_id);
create policy "own prefs" on user_preferences for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own memory notes" on user_memory_notes for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
-- query_cache and rate_limits intentionally get no policies -- see comments above.

-- ── Retrieval RPCs ────────────────────────────────────────────────────────
-- Hybrid retrieval calls both of these concurrently (ThreadPoolExecutor),
-- then fuses the two ranked lists with Reciprocal Rank Fusion.

create or replace function match_chunks(
  query_embedding vector,
  match_count integer,
  similarity_threshold double precision default 0.3,
  p_user_id uuid default null,
  p_document_ids uuid[] default null
)
returns table(content text, similarity double precision, filename text, page_number integer, chunk_index integer)
language sql stable as $$
  select content, 1 - (embedding <=> query_embedding) as similarity,
         filename, page_number, chunk_index
  from chunks
  where
    (p_user_id is null or user_id = p_user_id)
    and (p_document_ids is null or document_id = any(p_document_ids))
    and 1 - (embedding <=> query_embedding) > similarity_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;

create or replace function keyword_search_chunks(
  query_text text,
  match_count integer,
  p_user_id uuid default null,
  p_document_ids uuid[] default null
)
returns table(content text, rank double precision, filename text)
language sql stable as $$
  select chunks.content,
         ts_rank(content_tsv, query)::float as rank,
         chunks.filename
  from chunks, websearch_to_tsquery('english', query_text) query
  where
    content_tsv @@ query
    and (p_user_id is null or user_id = p_user_id)
    and (p_document_ids is null or document_id = any(p_document_ids))
  order by rank desc
  limit match_count;
$$;

-- ── Rate limiting + token usage RPCs ─────────────────────────────────────
-- Both share the rate_limits table via a windowed-counter pattern:
-- increment-and-return, resetting the window if it's expired.

create or replace function check_rate_limit(p_key text, p_limit integer, p_window_seconds integer)
returns boolean
language plpgsql as $$
declare
  v_count int;
begin
  insert into rate_limits (key, count, window_start)
  values (p_key, 1, now())
  on conflict (key) do update set
    count = case
      when now() - rate_limits.window_start > (p_window_seconds || ' seconds')::interval
        then 1
        else rate_limits.count + 1
    end,
    window_start = case
      when now() - rate_limits.window_start > (p_window_seconds || ' seconds')::interval
        then now()
        else rate_limits.window_start
    end
  returning count into v_count;

  return v_count <= p_limit;
end;
$$;

create or replace function increment_token_usage(p_key text, p_tokens integer, p_window_seconds integer)
returns integer
language plpgsql as $$
declare
  v_count int;
begin
  insert into rate_limits (key, count, window_start)
  values (p_key, p_tokens, now())
  on conflict (key) do update set
    count = case
      when now() - rate_limits.window_start > (p_window_seconds || ' seconds')::interval
        then p_tokens
        else rate_limits.count + p_tokens
    end,
    window_start = case
      when now() - rate_limits.window_start > (p_window_seconds || ' seconds')::interval
        then now()
        else rate_limits.window_start
    end
  returning count into v_count;
  return v_count;
end;
$$;
