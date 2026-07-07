import os
import time
from rag.db import get_supabase
from dotenv import load_dotenv

load_dotenv()

# get_full_graph/get_related_nodes each used to re-scan graph_nodes and
# graph_edges from Supabase in full on every single call -- opening the
# graph panel and clicking a topic both paid that cost even when nothing
# had changed since the last read. Cache the raw rows per user instead;
# add_to_graph() invalidates it after writing, so a read is never more
# than one write-cycle stale, and the graph is still built upon
# incrementally (new entities/edges append via add_to_graph as before) --
# only the *read* path stops redoing the full fetch every time.
_GRAPH_CACHE_TTL_SECONDS = 60
_graph_cache: dict[str, tuple[dict, float]] = {}


def _get_graph_data(user_id: str) -> dict:
    cached = _graph_cache.get(user_id)
    now = time.monotonic()
    if cached and cached[1] > now:
        return cached[0]

    supabase = get_supabase()
    nodes = supabase.table("graph_nodes").select("node_id, sources")\
        .eq("user_id", user_id).execute().data
    edges = supabase.table("graph_edges").select("source, target, relation")\
        .eq("user_id", user_id).execute().data

    data = {"nodes": nodes, "edges": edges}
    _graph_cache[user_id] = (data, now + _GRAPH_CACHE_TTL_SECONDS)
    return data


def invalidate_graph_cache(user_id: str) -> None:
    _graph_cache.pop(user_id, None)


def add_to_graph(extracted: dict, user_id: str):
    supabase = get_supabase()
    source = extracted.get("source", "unknown")

    for entity in extracted.get("entities", []):
        entity = entity.lower().strip()
        if not entity:
            continue

        existing = (
            supabase.table("graph_nodes")
            .select("id, sources")
            .eq("node_id", entity)
            .eq("user_id", user_id)
            .execute()
        )

        if existing.data:
            current_sources = existing.data[0].get("sources", [])
            if source not in current_sources:
                current_sources.append(source)
                supabase.table("graph_nodes").update(
                    {"sources": current_sources}
                ).eq("node_id", entity).eq("user_id", user_id).execute()
        else:
            supabase.table("graph_nodes").insert({
                "node_id": entity,
                "sources": [source],
                "user_id": user_id
            }).execute()

    for rel in extracted.get("relationships", []):
        subject = rel.get("subject", "").lower().strip()
        relation = rel.get("relation", "").lower().strip()
        obj = rel.get("object", "").lower().strip()

        if not subject or not relation or not obj:
            continue

        for node in [subject, obj]:
            exists = (
                supabase.table("graph_nodes")
                .select("id")
                .eq("node_id", node)
                .eq("user_id", user_id)
                .execute()
            )
            if not exists.data:
                supabase.table("graph_nodes").insert({
                    "node_id": node,
                    "sources": [source],
                    "user_id": user_id
                }).execute()

        existing_edge = (
            supabase.table("graph_edges")
            .select("id")
            .eq("source", subject)
            .eq("target", obj)
            .eq("user_id", user_id)
            .execute()
        )
        if not existing_edge.data:
            supabase.table("graph_edges").insert({
                "source": subject,
                "target": obj,
                "relation": relation,
                "user_id": user_id
            }).execute()

    invalidate_graph_cache(user_id)

def get_related_nodes(topic: str, user_id: str, depth: int = 2) -> dict:
    topic_lower = topic.lower().strip()
    topic_words = [w for w in topic_lower.split() if len(w) > 3]

    graph_data = _get_graph_data(user_id)
    all_nodes = graph_data["nodes"]
    all_edges = graph_data["edges"]

    matching = []
    for n in all_nodes:
        if topic_lower == n["node_id"]:
            matching.append(n["node_id"])

    if not matching:
        for n in all_nodes:
            if topic_lower in n["node_id"] or n["node_id"] in topic_lower:
                matching.append(n["node_id"])

    if not matching:
        for n in all_nodes:
            for word in topic_words:
                if word in n["node_id"]:
                    matching.append(n["node_id"])
                    break

    if not matching:
        return {"nodes": [], "edges": []}

    visited = {matching[0]}
    current = {matching[0]}

    for _ in range(depth):
        next_level = set()
        for node in current:
            for edge in all_edges:
                if edge["source"] == node:
                    next_level.add(edge["target"])
                if edge["target"] == node:
                    next_level.add(edge["source"])
        visited.update(next_level)
        current = next_level

    node_map = {n["node_id"]: n["sources"] for n in all_nodes}
    nodes = [{"id": n, "sources": node_map.get(n, [])} for n in visited]
    edges = [e for e in all_edges if e["source"] in visited and e["target"] in visited]

    return {"nodes": nodes, "edges": edges}

def get_full_graph(user_id: str) -> dict:
    graph_data = _get_graph_data(user_id)
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    return {
        "nodes": [{"id": n["node_id"], "sources": n["sources"]} for n in nodes],
        "edges": edges
    }