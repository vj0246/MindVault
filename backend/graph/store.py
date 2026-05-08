import json
import os
import networkx as nx
from networkx.readwrite import json_graph

GRAPH_FILE = "data/graph.json"


def _load_graph() -> nx.DiGraph:
    if os.path.exists(GRAPH_FILE):
        try:
            with open(GRAPH_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return nx.DiGraph()
                data = json.loads(content)
            return json_graph.node_link_graph(data, directed=True, edges="links")
        except Exception:
            return nx.DiGraph()
    return nx.DiGraph()


def _save_graph(G: nx.DiGraph):
    data = json_graph.node_link_data(G)
    with open(GRAPH_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_to_graph(extracted: dict):
    G = _load_graph()
    source = extracted.get("source", "unknown")

    # Add entity nodes
    for entity in extracted.get("entities", []):
        entity = entity.lower().strip()
        if not entity:
            continue
        if not G.has_node(entity):
            G.add_node(entity, sources=[source])
        else:
            existing = G.nodes[entity].get("sources", [])
            if source not in existing:
                existing.append(source)
                G.nodes[entity]["sources"] = existing

    # Add relationship edges
    for rel in extracted.get("relationships", []):
        subject = rel.get("subject", "").lower().strip()
        relation = rel.get("relation", "").lower().strip()
        obj = rel.get("object", "").lower().strip()

        if not subject or not relation or not obj:
            continue

        if not G.has_node(subject):
            G.add_node(subject, sources=[source])
        if not G.has_node(obj):
            G.add_node(obj, sources=[source])

        G.add_edge(subject, obj, relation=relation, source=source)

    _save_graph(G)
    print(f"[Graph] Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")


def get_related_nodes(topic: str, depth: int = 2) -> dict:
    G = _load_graph()
    topic_lower = topic.lower().strip()

    # Split into words and try to match any word against node IDs
    topic_words = [w for w in topic_lower.split() if len(w) > 3]
    
    matching = []
    
    # First try exact match
    for node in G.nodes:
        if topic_lower == node.lower():
            matching.append(node)
    
    # Then try contains match
    if not matching:
        for node in G.nodes:
            if topic_lower in node.lower() or node.lower() in topic_lower:
                matching.append(node)
    
    # Then try word by word
    if not matching:
        for node in G.nodes:
            for word in topic_words:
                if word in node.lower():
                    matching.append(node)
                    break

    if not matching:
        return {"nodes": [], "edges": []}

    start = matching[0]
    visited = set([start])
    current = {start}

    for _ in range(depth):
        next_level = set()
        for node in current:
            neighbors = set(G.successors(node)) | set(G.predecessors(node))
            next_level.update(neighbors)
        visited.update(next_level)
        current = next_level

    nodes = [{"id": n, "sources": G.nodes[n].get("sources", [])} for n in visited]
    edges = [
        {"source": u, "target": v, "relation": G.edges[u, v].get("relation", "related to")}
        for u, v in G.edges()
        if u in visited and v in visited
    ]

    return {"nodes": nodes, "edges": edges}


def get_full_graph() -> dict:
    G = _load_graph()
    return {
        "nodes": [
            {"id": n, "sources": G.nodes[n].get("sources", [])}
            for n in G.nodes
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "relation": G.edges[u, v].get("relation", "related to")
            }
            for u, v in G.edges()
        ]
    }