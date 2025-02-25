from extract_graph import naive_extract_graph
from build_tree import sequential_merge
from typing import List, Tuple, Dict
from itertools import combinations
import networkx as nx
import faiss
import spacy
from collections import defaultdict
from transformers import AutoTokenizer

class Retriever:
    def __init__(self, cache_tree, G:nx.Graph, index, nlp:spacy.Language, **kwargs) -> None:
        # index is the noun to chunks index.
        self.cache_tree = cache_tree
        self.G = G
        self.index = index
        self.nlp = nlp
        self.merge_num = kwargs.get("merge_num", 5)
        self.min_count = kwargs.get("min_count", 2)
        self.overlap = kwargs.get("overlap", 100)
        self.tokenizer = kwargs.get("tokenizer")
        print(self.tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
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
        chunk_ids = {}
        
        for entity in entities:
            if isinstance(entity, str):
                if entity in self.index.keys():
                    chunk_ids[entity] = self.index[entity]
            elif isinstance(entity, tuple):
                chunk_ids_set = set()
                entity_key = "_".join(entity)
                for e in entity:
                    if e in self.index.keys():
                        
                        if chunk_ids_set == set():
                            chunk_ids_set = set(self.index[e])
                        else:
                            chunk_ids_set = chunk_ids_set & set(self.index[e])
                chunk_ids[entity_key] = list(chunk_ids_set)

        return chunk_ids
    
    def get_shortest_path(self, entities:List[str], k) -> List[str]:
        # get the shortest path between the entities.
        shortest_path_pairs = []
        for head, tail in combinations(entities, 2):
            if head in self.G.nodes() and tail in self.G.nodes():
                try:
                    shortest_path = nx.shortest_path(self.G, head, tail)
                except nx.NetworkXNoPath:
                    continue
                if len(shortest_path) <= k:
                    shortest_path_pairs.append((head, tail))

        # shortest_path_pairs = self.merge_tuples(shortest_path_pairs)
        return shortest_path_pairs

    def merge_tuples(self, lst):
        # 用字典来跟踪每个实体的连接
        graph = defaultdict(set)
        
        # 构建实体间的关系图
        for a, b in lst:
            graph[a].add(b)
            graph[b].add(a)
        
        # 用来存储已处理的元组
        visited = set()
        result = []
        
        # 遍历所有的实体，寻找联通的元组
        def dfs(entity, cluster):
            if entity in visited:
                return
            visited.add(entity)
            cluster.add(entity)
            for neighbor in graph[entity]:
                dfs(neighbor, cluster)
        
        # 遍历所有的元组，合并相关的实体
        for a, b in lst:
            if a not in visited:
                cluster = set()
                dfs(a, cluster)
                result.append(tuple(sorted(cluster)))
        
        return result
    
    def validate_by_checking_father_chunks(self, init_chunk_ids:Dict[str, List[str]]) -> Dict[str, List[str]]:
        # w, by input the shortest path pairs, get the father nodes.
        valid_child_ids = {}
        for key, chunk_ids in init_chunk_ids.items():
            father_nodes = {}
            for chunk_id in chunk_ids:
                father_chunk_id = self.cache_tree[chunk_id]["parent"]
                father_nodes.setdefault(father_chunk_id, []).append(chunk_id)
            valid_leaf_nodes = [child_id_list for _, child_id_list in father_nodes.items()
                               if len(child_id_list) >= self.min_count]
            
            valid_child_ids[key] = []
            if len(valid_leaf_nodes) > 0:
                for leaf_node in valid_leaf_nodes:
                    valid_child_ids[key].extend(leaf_node)
        
        return valid_child_ids

    # def get_leaf_chunks(self, valid_father_nodes:List[str]) -> List[str]:
    #     # get the leaf chunks from the father nodes.
    #     # get the direct children of the father nodes.
    #     leaf_nodes = []
    #     for father_node in valid_father_nodes:
    #         children = self.cache_tree[father_node]["children"]
    #         leaf_nodes.extend([child for child in children if child in ])

    def _detect_neighbor_nodes(self, keys:List[str], chunk_id: str) -> List[str]:
        # detect the neighbor nodes of the chunk_id.
        # return the neighbor nodes.
        int_chunk_id = int(chunk_id.split("_")[-1])
        front = True
        back = True
        neighbor_nodes = [chunk_id]
        front_int_chunk_id = int_chunk_id
        back_int_chunk_id = int_chunk_id
        while front or back:
            if front:
                front_int_chunk_id = front_int_chunk_id - 1
                if front_int_chunk_id < 0 or "leaf_{}".format(front_int_chunk_id) not in keys:
                    front = False
                str_chunk_id = "leaf_{}".format(front_int_chunk_id)
                append = True
                for key in keys:
                    if str_chunk_id not in self.index.get(key, []):
                        append = False
                        front = False
                        break
                if append:
                    neighbor_nodes.append(str_chunk_id)
            if back:
                back_int_chunk_id += 1
                if "leaf_{}".format(back_int_chunk_id) not in keys:
                    back = False
                str_chunk_id = "leaf_{}".format(back_int_chunk_id)
                append = True
                for key in keys:
                    if str_chunk_id not in self.index.get(key, []):
                        append = False
                        back = False
                        break
                if append:
                    neighbor_nodes.append(str_chunk_id)
        return neighbor_nodes

    def get_neighbor_chunks(self, leaf_nodes:List[str]) -> List[str]:
        # get the neighbor chunks from the leaf nodes.
        all_neighbor_nodes = {}
        for keys, chunk_ids in leaf_nodes.items():
            key_list = list(keys.split("_"))
            for chunk_id in chunk_ids:
                neighbor_nodes = self._detect_neighbor_nodes(key_list, chunk_id)
                all_neighbor_nodes.setdefault(keys, []).extend(neighbor_nodes)
        for key, chunk_lists in all_neighbor_nodes.items():
            all_neighbor_nodes[key] = sorted(list(set(chunk_lists)))
        return all_neighbor_nodes

    def merge_keys(self, neighbor_nodes:Dict[str, List[str]]) -> Dict[str, List[str]]:
        # merge the nodes with different keys.
        # return with the same format.
        chunks_to_keys = defaultdict(set)
        for key, chunk_lists in neighbor_nodes.items():
            for chunk in chunk_lists:
                chunks_to_keys[chunk].add(key)

        merged_result = {}
        for chunk, keys in chunks_to_keys.items():
            # get the new key.
            if len(keys) > 1:
                all_entities = set()
                for key in keys:
                    all_entities.update(key.split("_"))
                new_key = "_".join(sorted(all_entities))
            else:
                new_key = keys.pop()
            # add the chunk to the new key.
            if new_key in merged_result.keys():
                merged_result.setdefault(new_key, []).append(chunk)
            else:
                merged_result[new_key] = [chunk]
        return merged_result


    def get_contiguous_chunks(self, leaf_nodes:List[str]) -> str:
        leaf_texts = []
        for leaf_node in leaf_nodes:
            leaf_text = self.cache_tree[leaf_node]["text"]
            leaf_texts.append(leaf_text)
        return sequential_merge(leaf_texts, self.tokenizer, self.overlap)

    def format_res(self, res:Dict[str, List[str]]) -> str:
        res_str = ""
        for key, chunks in res.items():
            res_str += "{}: {}\n".format(key, chunks)
        return res_str

    def query(self, query, **kwargs):
        
        entities = naive_extract_graph(query.split("\n")[0], self.nlp)

        entities = entities["nouns"]

        if kwargs.get("wasd", True):
            # initialize by shortest path
            shortest_path = self.get_shortest_path(entities, kwargs.get("shortest_path_k", kwargs.get("shortest_path_k", 4)))
            
            # initialize the chunks.
            init_chunk_ids = self.get_chunks(shortest_path)
            
            # w+s step, get the valid chunks by checking the father of the leaves.
            valid_child_ids = self.validate_by_checking_father_chunks(init_chunk_ids)
            # print("valid_child_ids", valid_child_ids)
            # a+d step, search the neighbor of the leaf nodes.
            neighbor_nodes = self.get_neighbor_chunks(valid_child_ids)
            # print("neighbor_nodes", neighbor_nodes)
            # merge the nodes with different keys.
            neighbor_nodes = self.merge_keys(neighbor_nodes)
            print("merged neighbor_nodes", neighbor_nodes)

            res = {}
            chunk_count = 0
            for key, chunk_ids in neighbor_nodes.items():
                res[key] = chunk_ids
                chunk_count += len(chunk_ids)
            res_str = self.format_res(res)

            result = {"chunks":res_str}
            if kwargs.get("debug", False):
                result["entities"] = entities
                result["neighbor_nodes"] = neighbor_nodes
                result["keys"] = list(neighbor_nodes.keys())
                result["len_chunks"] = chunk_count
            return result

        if kwargs.get("related_entities", False):
            # get related entities.
            entities = self.get_related_entities(entities)

        if kwargs.get("shortest_path", True):
            # get the shortest path between the entities.
            shortest_path = self.get_shortest_path(entities, kwargs.get("shortest_path_k", kwargs.get("shortest_path_k", 4)))
            entities = shortest_path

        # use the entities to get the chunks
        chunks = self.get_chunks(entities)
    
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
        

        res_str = "\n".join(chunks)
        result = {"chunks":res_str}

        if kwargs.get("debug", False):
            result["entities"] = entities
            result["len_chunks"] = len(chunks)
        return result
        