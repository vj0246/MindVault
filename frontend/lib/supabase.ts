import { createClient, SupabaseClient } from '@supabase/supabase-js'

let _client: SupabaseClient | null = null

export function getSupabase(): SupabaseClient {
  if (typeof window === 'undefined') {
    throw new Error('Supabase client only available in browser')
  }
  if (!_client) {
    _client = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    )
  }
  return _client
}

export async function getToken(): Promise<string | null> {
  if (typeof window === 'undefined') return null
  const { data: { session } } = await getSupabase().auth.getSession()
  return session?.access_token || null
}

export async function signOut() {
  await getSupabase().auth.signOut()
  window.location.href = '/login'
}