from extract_graph import naive_extract_graph
from typing import List
from itertools import combinations
import networkx as nx
import faiss
import spacy

class Retriever:
    def __init__(self, cache_tree, G:nx.Graph, index, nlp:spacy.Language, **kwargs) -> None:
        # index is the noun to chunks index.
        self.cache_tree = cache_tree
        self.G = G
        self.index = index
        self.nlp = nlp
        if kwargs.get("embedder", None) is not None:
            self.embedder = kwargs.get("embedder")
            self.faiss_index, self.docs = self._build_faiss_index()
        else:
            self.embedder = None
            self.faiss_index = None

    def _build_faiss_index(self):
        # build the faiss index.
        # only used when the dense retrieval is implemented.
        # return the faiss index.
        docs = []
        for key, value in self.cache_tree.items():
            docs.append(value["text"])
        
        # use the embedder to embed the docs.
        # use the faiss to build the index.
        # return the faiss index.
        doc_embeds = self.embedder(docs)
        vector_database = faiss.IndexFlatIP(doc_embeds.shape[1])
        vector_database.add(doc_embeds)
        return vector_database, docs

    def get_related_entities(self, entities:List[str]) -> List[str]:
        related_entities = []
        for source, target in combinations(entities, 2):
            if source in self.G.nodes() and target in self.G.nodes():
                shortest_path = nx.shortest_path(self.G, source, target)
                for node in shortest_path:
                    if node not in related_entities:
                        related_entities.append(node)
        return related_entities

    def get_chunks(self, entities:List[str]) -> List[str]:
        # get the chunks from the cache tree.
        chunk_ids = []
        # TODO:这里是把所有的entity都加进去了，需要优化。
        for entity in entities:
            if entity in self.index.keys():
                chunk_ids.extend(self.index[entity])
        
        chunks = []
        for chunk_id in chunk_ids:
            chunks.append(self.cache_tree[chunk_id]["text"])
        return chunks

    def query(self, query, **kwargs):
        
        entities = naive_extract_graph(query, self.nlp)

        entities = entities["nouns"]

        if kwargs.get("related_entities", False):
            # get related entities.
            related_entities = self.get_related_entities(entities)

        # use the entities to get the chunks
        chunks = self.get_chunks(self.cache_tree, self.index, entities)
    
        # search for the other chunks.
        # TODO: using faiss to search for the other chunks. maybe ablation study.

        if kwargs.get("dense_retrieval", False):
            # using dense retrieval to get the other chunks.
            assert self.embedder is not None, "The embedder is not set, please set the embedder when init the retriever."
            assert self.faiss_index is not None, "The faiss index is not set."
            query_embed = self.embedder(query)
            dense_chunks = self.faiss_index.search(query_embed, k=kwargs.get("k", 5))
            dense_chunks = [self.docs[i] for i in dense_chunks]
            chunks.extend(dense_chunks)
        
        result = {"chunks":chunks}

        if kwargs.get("debug", False):
            result["entities"] = entities
            result["related_entities"] = related_entities

        return result
        