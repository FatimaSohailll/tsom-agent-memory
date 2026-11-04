"""NetworkX view over the SQLite store, for graph queries and plots."""

import networkx as nx

from .store import Store


def to_graph(store: Store) -> nx.DiGraph:
    g = nx.DiGraph()
    for note in store.list_notes():
        g.add_node(note.id, content=note.content, timestamp=note.timestamp)
        for target in note.links:
            g.add_edge(note.id, target)
    return g
