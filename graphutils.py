''' this file mainly for:
        1. merge the entities inside the graph
        2. get the shortest path inside the graph
        3. convert the books into networkx.
'''

import networkx as nx
import json
from itertools import combinations
from typing import List, Tuple

def build_graph(triplets:List[Tuple[str, str, int]]) -> nx.Graph:
    '''
    build the graph from the triplets.
    '''
    G = nx.Graph()
    for triplet in triplets:
        G.add_edge(triplet[0], triplet[1], weight=triplet[2])
    return G

def get_shortest_path(G:nx.Graph, start:str, end:str) -> List[str]:
    '''
    get the shortest path between start and end.
    '''
    return nx.shortest_path(G, start, end, weight='weight')

def multi_shortest_path(G:nx.Graph, entities:List[str]) -> List[List[str]]:
    '''
    get the shortest path between all entities.
    '''
    paths = []
    for i, j in combinations(entities, 2):
        path = get_shortest_path(G, i, j)
        paths.append(path)
    return paths
