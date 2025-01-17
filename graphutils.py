''' this file mainly for:
        1. merge the entities inside the graph
        2. get the shortest path inside the graph
        3. convert the books into networkx.
'''

import networkx as nx
import torch
from itertools import combinations
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cluster import DBSCAN

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

def merge_entities(triplets:List[Tuple[str, str, int]]) -> Tuple[List[Tuple[str, str, int]], dict]:
    '''
    merge the entities into the graph using sklearn clustering
    '''
    import transformers
    model = transformers.AutoModel.from_pretrained("bert-base-uncased").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    nodes = set([i[0] for i in triplets] + [i[1] for i in triplets])
    nodes = list(nodes)
    node_embeddings = []
    batch_size = 32
    for i in range(0, len(nodes), batch_size):
        batch_nodes = nodes[i:i + batch_size]
        tokenized = tokenizer(batch_nodes, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokenized)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        node_embeddings.extend(batch_embeddings.tolist())
    
    # Convert embeddings to numpy array
    node_embeddings = np.array(node_embeddings)
            
    clustering = DBSCAN(eps=0.1, min_samples=1, metric='cosine')
    cluster_labels = clustering.fit_predict(node_embeddings)

    clusters = {}
    for node, label in zip(nodes, cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node)
    
    node_name_mapping = {}
    for label, members in clusters.items():
        if label == -1:
            for member in members:
                node_name_mapping[member] = member
        else:
            rep = max(members, key=len)
            for member in members:
                node_name_mapping[member] = rep

    merged_triplets = []
    for s, t, w in triplets:
        new_s = node_name_mapping.get(s, s)
        new_t = node_name_mapping.get(t, t)
        if new_s != new_t:
            merged_triplets.append((new_s, new_t, w))
    
    return merged_triplets, node_name_mapping
    
    
