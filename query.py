from extract_graph import naive_extract_graph
from typing import List
from itertools import combinations
import networkx as nx
# retriever should be init with cache_tree and G : nx.netxxx
# also index.

class Retriever:
    def __init__(self, cache_tree, G:nx.Graph, index, nlp) -> None:
        self.cache_tree = cache_tree
        self.G = G
        self.index = index
        self.nlp = nlp

    def get_related_entities(self, entities:List[str]) -> List[str]:
        related_entities = []
        for source, target in combinations(entities, 2):
            if source in self.G.nodes() and target in self.G.nodes():
                shortest_path = nx.shortest_path(self.G, source, target)
                for node in shortest_path:
                    if node not in related_entities:
                        related_entities.append(node)
        return related_entities


    def query(self, query):
        entities = naive_extract_graph(query, self.nlp)

        entities = entities["nouns"]

        # # get related entities.
        # related_entities = self.get_related_entities(entities)

        # use the entities to get the chunks
        chunks = self.cache_tree.get_chunks(entities)
    
        # search for the other chunks.
        # TODO: using faiss to search for the other chunks. maybe ablation study.

        