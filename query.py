from extract_graph import naive_extract_graph
from build_tree import sequential_merge
from typing import List, Tuple, Dict, Set
from itertools import combinations
import networkx as nx
import faiss
import spacy
from collections import defaultdict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import random
random.seed(1)
import copy
import numpy as np

class Retriever:
    def __init__(self, cache_tree, G:nx.Graph, index, appearance_count:Dict[str, int], nlp:spacy.Language, **kwargs) -> None:
        # index is the noun to chunks index.
        # appearance_count is the appearance count of the entities in the chunks.
        self.cache_tree = cache_tree
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(self.cache_tree)
        self.G = G
        self.index = index
        self.appearance_count = appearance_count
        self.inverse_index = self.get_inverse_index()
        self.nlp = nlp
        self.device = kwargs.get("device", "cuda:0")
        self.merge_num = kwargs.get("merge_num", 5)
        self.min_count = kwargs.get("min_count", 2)
        self.overlap = kwargs.get("overlap", 100)
        self.tokenizer = kwargs.get("tokenizer","/root/shared_planing/LLM_model/Qwen2.5-7B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        if kwargs.get("embedder", "BAAI/bge-m3") is not None:
            self.embedder = SentenceTransformer(kwargs.get("embedder", "BAAI/bge-m3"),device=self.device)
            self.faiss_index = self._build_faiss_index()
        else:
            print("Warning: the embedder is set to None, dense retrieval is not implemented.")
            self.embedder = None
            self.faiss_index = None

    def __del__(self):
        """Ensure proper cleanup of resources"""
        try:
            if hasattr(self, 'embedder'):
                del self.embedder
            if hasattr(self, 'faiss_index'):
                del self.faiss_index
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during Retriever cleanup: {e}")

    def update(self, cache_tree, G, index, appearance_count):
        self.cache_tree = cache_tree
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(self.cache_tree)
        self.G = G
        self.index = index
        self.appearance_count = appearance_count
        self.inverse_index = self.get_inverse_index()
        self.docs = self.collapse_tree
        if self.embedder is not None:
            self.faiss_index = self._build_faiss_index()

    def get_inverse_index(self):
        # get the inverse index.
        inverse_index = {}
        for key, value in self.index.items():
            for chunk_id in value:
                inverse_index.setdefault(chunk_id, []).append(key)
        return inverse_index

    def _collapse_tree(self, cache_tree:Dict[str, Dict]) -> Dict[str, Dict]:
        # collapse the tree.
        # return the collapsed tree.
        collapsed_tree = []
        collapsed_tree_ids = []
        for key, value in self.cache_tree.items():
            collapsed_tree.append(value["text"])
            collapsed_tree_ids.append(key)
        return collapsed_tree, collapsed_tree_ids

    def _build_faiss_index(self):
        # build the faiss index.
        # only used when the dense retrieval is implemented.
        # return the faiss index.
        docs = self.collapse_tree
        if self.embedder is None:
            self.embedder = SentenceTransformer("BAAI/bge-m3",device=self.device)
            self.embedder.eval()
            print("the embedder is not set, using the default embedder BAAI/bge-m3.")
        doc_embeds = self.embedder.encode(docs, batch_size=16, device=self.device)
        # print("doc_embeds examples", doc_embeds[0:5][0:5])
        # print("doc_embeds shape", doc_embeds.shape)
        vector_database = faiss.IndexFlatIP(doc_embeds.shape[1])
        vector_database.add(doc_embeds)
        return vector_database

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
                chunk_ids[entity_key] = sorted(list(chunk_ids_set))

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
        graph = defaultdict(set)
        
        for a, b in lst:
            graph[a].add(b)
            graph[b].add(a)
        
        visited = set()
        result = []
        
        def dfs(entity, cluster):
            if entity in visited:
                return
            visited.add(entity)
            cluster.add(entity)
            for neighbor in graph[entity]:
                dfs(neighbor, cluster)
        
        for a, b in lst:
            if a not in visited:
                cluster = set()
                dfs(a, cluster)
                result.append(tuple(sorted(cluster)))
        
        return result
    
    def validate_by_checking_father_chunks(self, init_chunk_ids:Dict[str, List[str]], min_count:int=2) -> Dict[str, List[str]]:
        # w, by input the shortest path pairs, get the father nodes.
        valid_child_ids = {}
        for key, chunk_ids in init_chunk_ids.items():
            father_nodes = {}
            for chunk_id in chunk_ids:
                father_chunk_id = self.cache_tree[chunk_id]["parent"]
                father_nodes.setdefault(father_chunk_id, []).append(chunk_id)
            valid_leaf_nodes = [child_id_list for _, child_id_list in father_nodes.items()
                               if len(child_id_list) >= min_count]
            
            valid_child_ids[key] = []
            if len(valid_leaf_nodes) > 0:
                for leaf_node in valid_leaf_nodes:
                    valid_child_ids[key].extend(leaf_node)
        
        return valid_child_ids

    # def _detect_neighbor_nodes(self, keys:Set[str], chunk_id: str) -> List[str]:
    #     # detect the neighbor nodes of the chunk_id.
    #     # return the neighbor nodes.
    #     int_chunk_id = int(chunk_id.split("_")[-1])
    #     front = True
    #     back = True
    #     neighbor_nodes = [chunk_id]
    #     front_int_chunk_id = int_chunk_id
    #     back_int_chunk_id = int_chunk_id
    #     while front or back:
    #         if front:
    #             front_int_chunk_id = front_int_chunk_id - 1
    #             if front_int_chunk_id < 0 or set(self.inverse_index.get(front_int_chunk_id, [])) & keys != keys:
    #                 front = False
    #             str_chunk_id = "leaf_{}".format(front_int_chunk_id)
    #             append = True
    #             for key in keys:
    #                 if str_chunk_id not in self.index.get(key, []):
    #                     append = False
    #                     front = False
    #                     break
    #             if append:
    #                 neighbor_nodes.append(str_chunk_id)
    #         if back:
    #             back_int_chunk_id += 1
    #             if set(self.inverse_index.get(back_int_chunk_id, [])) & keys != keys:
    #                 back = False
    #             str_chunk_id = "leaf_{}".format(back_int_chunk_id)
    #             append = True
    #             for key in keys:
    #                 if str_chunk_id not in self.index.get(key, []):
    #                     append = False
    #                     back = False
    #                     break
    #             if append:
    #                 neighbor_nodes.append(str_chunk_id)
    #     return neighbor_nodes

    def get_neighbor_chunks(self, leaf_nodes:List[str]) -> List[str]:
        # get the neighbor chunks from the leaf nodes.
        all_neighbor_nodes = {}
        for keys, chunk_ids in leaf_nodes.items():
            key_set = set(keys.split("_"))
            for chunk_id in chunk_ids:
                neighbor_nodes = self._detect_neighbor_nodes(key_set, chunk_id)
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

    def detect_contiguous_chunks(self, chunk_ids:List[str]) -> List[List[str]]:
        # Detect the contiguous chunks from the chunk_ids,
        # if there are contiguous chunks, return the list of the contiguous chunks.
        # otherwise, the only id will be a list.
        res = []
        current_chunk = []
        chunk_ids = sorted(chunk_ids, key=lambda x: int(x.split("_")[1]))
        for chunk_id in chunk_ids:
            # Extract the numeric part of the chunk_id
            id_num = int(chunk_id.split("_")[1])
            if not current_chunk:
                current_chunk.append(chunk_id)
            else:
                # Check if the current id is contiguous with the last one
                last_id_num = int(current_chunk[-1].split("_")[1])
                if id_num == last_id_num + 1:
                    current_chunk.append(chunk_id)
                else:
                    res.append(current_chunk)
                    current_chunk = [chunk_id]

        if current_chunk:
            res.append(current_chunk)

        return res

    def format_res(self, res:Dict[str, List[str]]) -> str:
        res_str = ""
        for key, chunks in res.items():
            chunks = self.detect_contiguous_chunks(chunks)
            for chunk_list in chunks:
                str_of_list = self.get_contiguous_chunks(chunk_list)
                res_str += "{}: {}\n".format(key, str_of_list)
        return res_str
    
    def str_chunkid_2_int_chunkid(self, str_chunk:str) -> int:
        return int(str_chunk.split("_")[-1])

    def wasd_step(self, entities:List[str], shortest_path_k:int=4, min_count:int=2)->Dict[str, List[str]]:
        # initialize by shortest path
        shortest_path = self.get_shortest_path(entities, shortest_path_k) 
        # it returns the list of pairs existing shortest path shorter than k.
        
        # initialize the chunks.
        init_chunk_ids = self.get_chunks(shortest_path)
        # it returns a dict, key is entityA_entityB, value is the sorted list of chunk ids.
        ##################### tree filter, not work ################
        # # w+s step, get the valid chunks by checking the father of the leaves.
        # valid_child_ids = self.validate_by_checking_father_chunks(init_chunk_ids, min_count)
        # # it won't change the data structure, just filter the chunks.

        # # a+d step, search the neighbor of the leaf nodes.
        # neighbor_nodes = self.get_neighbor_chunks(valid_child_ids)
        # # won't change the data structure, just add the chunks.
        
        # merge the nodes with different keys, the structure is still the same.
        neighbor_nodes = self.merge_keys(init_chunk_ids)
        return neighbor_nodes

    def wasd_ablation(self, entities:List[str], shortest_path_k:int=4, min_count:int=2, abtype = "")->Dict[str, List[str]]:
        if abtype == "wo_graphfilter" or abtype == "wo_GF&EAR":
            shortest_path_pairs = []
            for entity1, entity2 in combinations(entities, 2):
                shortest_path_pairs.append((entity1, entity2))
            init_chunk_ids = self.get_chunks(shortest_path_pairs)
            neighbor_nodes = self.merge_keys(init_chunk_ids)
            return neighbor_nodes

    #     if abtype == "wo_treefilter":
    #         shortest_path = self.get_shortest_path(entities, shortest_path_k) 
    #         # it returns the list of pairs existing shortest path shorter than k.
    #         init_chunk_ids = self.get_chunks(shortest_path)
    #         init_chunk_ids = self.merge_keys(init_chunk_ids)
    #         return init_chunk_ids
        
    #     shortest_path = self.get_shortest_path(entities, shortest_path_k) 
    #     # it returns the list of pairs existing shortest path shorter than k.
        
    #     # initialize the chunks.
    #     init_chunk_ids = self.get_chunks(shortest_path)
    #     if abtype == "wo_fatherfilter":
    #         neighbor_nodes = self.get_neighbor_chunks(init_chunk_ids)
    #         # won't change the data structure, just add the chunks.
            
    #         # merge the nodes with different keys, the structure is still the same.
    #         neighbor_nodes = self.merge_keys(neighbor_nodes)
    #         return neighbor_nodes
    #     elif abtype == "wo_neighbordetection":
    #         valid_child_ids = self.validate_by_checking_father_chunks(init_chunk_ids, min_count)
    #         neighbor_nodes = self.merge_keys(valid_child_ids)
    #         return neighbor_nodes
    #     else:
    #         raise NotImplementedError("The ablation type {} is not implemented.".format(abtype))
    # # def filter_chunk_by_entities(self, condidate_chunk_ids:List[str], entities:List[str]) -> List[str]:
    # #     # filter the chunks that not contain the related entities.
    # #     filtered_chunk_ids = []
    # #     for chunk_id in condidate_chunk_ids:
    # #         if chunk_id in self.index.keys():
    # #             if set(self.index[chunk_id]) & set(entities):
    # #                 filtered_chunk_ids.append(chunk_id)
    # #     res = {}
    # #     for entity in entities:
    # #         res[entity] = filtered_chunk_ids
    # #     return res

    def dense_retrieval(self, query,k):
        # using dense retrieval to get the chunks.
        query_embed = self.embedder.encode(query).reshape(1, -1) # need (1, -1) for faiss.
        _, condidate_chunks_indexs = self.faiss_index.search(query_embed, k = k)
        # the normal faiss index return the (1, k) shape. squeeze it to (k,).
        condidate_chunks_indexs = condidate_chunks_indexs[0]
        condidate_chunk_ids = [self.collapse_tree_ids[i] for i in condidate_chunks_indexs]
        res = {"": condidate_chunk_ids}
        return res

    def _count_chunks(self, res:Dict[str, List[str]]) -> int:
        # count the chunks.
        count = 0
        for chunk_ids in res.values():
            count += len(chunk_ids)
        return count

    def entityaware_filter(self, candidate_chunks:Dict[str, List[str]], entities:List[str]) -> Dict[str, List[str]]:
        # filter rules:
        # 1. the chunk includes more different entities, the priority is higher.
        # 2. if the chunk has longer neighbor nodes, the priority is higher.
        # 3. if the chunk includes the same number of entities, the higher the number of appearance of the entities is, the higher the priority is.
        # Initialize result dictionary
        chunks_info = []
        for key, value in candidate_chunks.items():
            for chunk_id in value:
                key_count = len(key.split("_"))
                set_key = set(key.split("_"))
                neighbor_nodes_count = len(self._detect_neighbor_nodes(keys=set_key, chunk_id=chunk_id))
                entity_count = 0
                for key_entity in key.split('_'):
                    entity_count += self.appearance_count[chunk_id].get(key_entity, 0)
                chunk_id_info = {
                    "chunk_id": chunk_id,
                    "key_count": key_count,
                    "neighbor_nodes_count": neighbor_nodes_count,
                    "entity_count": entity_count
                }
                chunks_info.append(chunk_id_info)
        # sort the chunks_info by the key_count, neighbor_nodes_count, and entity_count.
        sorted_chunks_info = sorted(chunks_info, key=lambda x: (x["key_count"], x["neighbor_nodes_count"], x["entity_count"]), reverse=True)
        # get the top 25 chunks.
        top_25_chunks = sorted_chunks_info[:25]
        # get the chunk_ids from the top_25_chunks.
        top_25_chunk_ids = [chunk["chunk_id"] for chunk in top_25_chunks]
        # return the result.
        filtered_res = {}
        for id in top_25_chunk_ids:
            for entity in entities:
                if id in self.index.get(entity, []):
                    filtered_res.setdefault(entity, []).append(id)
        filtered_res = self.merge_keys(filtered_res)
        print("filtered_res,",filtered_res)
        return filtered_res

    def _check_children(self, chunk_id:str, entities:List[str], visited=None) -> int:
        # 添加visited集合来防止循环递归
        if visited is None:
            visited = set()
        
        # 如果当前chunk_id已经访问过，直接返回0避免重复计算
        if chunk_id in visited:
            return 0
        
        # 将当前chunk_id添加到已访问集合
        visited.add(chunk_id)
        
        # Initialize a counter for the number of entities found in the children
        entity_count = 0
        
        # Get the children of the chunk_id
        children = self.cache_tree.get(chunk_id, {}).get("children", [])

        # Iterate through each child and count the entities
        for child in children:
            if not child.startswith("leaf_"):
                entity_count += self._check_children(child, entities, visited)
            else:
                # 对于叶子节点，检查实体出现情况
                chunk_appearance_stat = self.appearance_count.get(child, {})
                for entity in entities:
                    entity_count += chunk_appearance_stat.get(entity, 0)
        
        return entity_count

    def _faiss_entity_filter(self, candidate_chunk_ids:List[str], entities:List[str]) -> Dict[str, List[str]]:
        # filter the chunks that not contain the related entities.
        filtered_res = {}
        filtered_chunk_ids = []
        chunk_count = []
        
        for chunk_id in candidate_chunk_ids:
            if not chunk_id.startswith("leaf_"):
                chunk_count.append(self._check_children(chunk_id, entities))
                continue
            
            chunk_appearance_stat = self.appearance_count.get(chunk_id, {})
            this_chunk_count = 0
            for entity in entities:
                this_chunk_count += chunk_appearance_stat.get(entity, 0)
            chunk_count.append(this_chunk_count)

        chunk_count = np.array(chunk_count)
        nonzero_indices = np.nonzero(chunk_count)[0]
        
        if len(nonzero_indices) == 0:
            return {"": candidate_chunk_ids[:25]}
        
        argsorted_chunk_ids = np.argsort(chunk_count)[::-1]
        filtered_chunk_ids = [candidate_chunk_ids[i] for i in argsorted_chunk_ids if chunk_count[i] > 0]
        
        if len(filtered_chunk_ids) > 25:
            filtered_chunk_ids = filtered_chunk_ids[:25]
        
        for id in filtered_chunk_ids:
            for entity in entities:
                if entity in self.inverse_index.get(id, []):
                    filtered_res.setdefault(entity, []).append(id)
        
        filtered_res = self.merge_keys(filtered_res)
        return filtered_res

    def _ablation_query(self, query, **kwargs):
        '''
        conduct the ablation study for the query.
        ablation_type:
        1. faiss only. means we only leverage the tree and the vector based retrieval.
        '''
        ablation_type = kwargs.get("ablation_type", "faiss_only")
        if ablation_type == "faiss_only":
            chunk_ids = self.dense_retrieval(query, kwargs.get("max_chunk_setting", 25))
            result = {"chunks":self.format_res(chunk_ids)}
            if kwargs.get("debug", True):
                supplement_info = self._build_supplement_info(chunk_ids, None, chunk_ids, list(chunk_ids.keys()), len(chunk_ids[""]), [])
                result.update(supplement_info)
            return result

        elif ablation_type == "wo_graphfilter" or ablation_type == "wo_treefilter" or ablation_type == "wo_neighbordetection" or ablation_type == "wo_fatherfilter":
            entities = naive_extract_graph(query.split("\n")[0], self.nlp)["nouns"]
            shortest_path_k = kwargs.get("shortest_path_k", 4)
            min_count = kwargs.get("min_count", 2)
            if len(entities) == 0:
                chunk_ids = self.dense_retrieval(query, kwargs.get("max_chunk_setting", 25))
                result = {"chunks":self.format_res(chunk_ids)}
                if kwargs.get("debug", True):
                    supplement_info = self._build_supplement_info(chunk_ids, entities, chunk_ids, list(chunk_ids.keys()), len(chunk_ids), [])
                    result.update(supplement_info)
                    result["is_global_retrieval"] = True
                return result
            wasd_res = self.wasd_ablation(entities, shortest_path_k, min_count, ablation_type)
            chunk_count = self._count_chunks(wasd_res)
            chunk_counts_history = []
            chunk_counts_history.append((shortest_path_k, min_count, chunk_count))

            if chunk_count == 0:
                query_embed = self.embedder.encode(query).reshape(1, -1) # need (1, -1) for faiss.
                _, condidate_chunks_indexs = self.faiss_index.search(query_embed, k = 25 *2)
                # the normal faiss index return the (1, k) shape. squeeze it to (k,).
                condidate_chunks_indexs = condidate_chunks_indexs[0]
                condidate_chunk_ids = [self.collapse_tree_ids[i] for i in condidate_chunks_indexs]
                filtered_chunk_ids = self._faiss_entity_filter(condidate_chunk_ids, entities) 
                # return the entity_entityB: [chunk_id1, chunk_id2, ...]
                res_str = self.format_res(filtered_chunk_ids)

                result = {"chunks":res_str}
                if kwargs.get("debug", True):
                    supplement_info = self._build_supplement_info(filtered_chunk_ids, entities, filtered_chunk_ids, list(filtered_chunk_ids.keys()), chunk_count, chunk_counts_history)
                    result.update(supplement_info)
                return result
            
            else:
                candidate_chunks = wasd_res
                res_ids = self.entityaware_filter(candidate_chunks, entities)
                chunk_count = self._count_chunks(res_ids)
                res_str = self.format_res(res_ids)
                result = {"chunks":res_str}
                result["chunk_counts_history"] = chunk_counts_history
                if kwargs.get("debug", True):
                    supplement_info = self._build_supplement_info(res_ids, entities, res_ids, list(res_ids.keys()), chunk_count, chunk_counts_history)
                    result.update(supplement_info)
                return result
            
        elif ablation_type == "wo_GR&OR":
            entities = naive_extract_graph(query.split("\n")[0], self.nlp)["nouns"]
            shortest_path_k = kwargs.get("shortest_path_k", 4)
            min_count = kwargs.get("min_count", 2)
            if len(entities) == 0:
                # NOTE: without global retrieval, the result is empty.
                return {"chunks":""}
            wasd_res = self.wasd_step(entities, shortest_path_k, min_count)
            chunk_count = self._count_chunks(wasd_res)
            chunk_counts_history = []
            chunk_counts_history.append((shortest_path_k, min_count, chunk_count))

            if chunk_count == 0:
                result = {"chunks":""}
                return result
            
            else:
                flag = 0
                while chunk_count > kwargs.get("max_chunk_setting", 25):
                    prev_wasd_res = copy.deepcopy(wasd_res)
                    if flag == 0:
                        shortest_path_k -= 1
                        flag = 1
                    else:
                        min_count += 1
                        flag = 0
                    wasd_res = self.wasd_step(entities, shortest_path_k, min_count)
                    chunk_count = self._count_chunks(wasd_res)
                    chunk_counts_history.append((shortest_path_k, min_count, chunk_count))
                if chunk_count != 0:
                    print("BOTTOM2TOP: final chunk_count", chunk_count)
                    res_str = self.format_res(wasd_res)

                    result = {"chunks":res_str}
                    if kwargs.get("debug", True):
                        supplement_info = self._build_supplement_info(wasd_res, entities, wasd_res, list(wasd_res.keys()), chunk_count, chunk_counts_history)
                        result.update(supplement_info)
                    return result
                else:
                    candidate_chunks = prev_wasd_res
                    res_ids = self.entityaware_filter(candidate_chunks, entities)
                    chunk_count = self._count_chunks(res_ids)
                    res_str = self.format_res(res_ids)
                    result = {"chunks":res_str}
                    result["chunk_counts_history"] = chunk_counts_history
                    if kwargs.get("debug", True):
                        supplement_info = self._build_supplement_info(res_ids, entities, res_ids, list(res_ids.keys()), chunk_count, chunk_counts_history)
                        result.update(supplement_info)
                    return result


        elif ablation_type == "wo_global_retrieval":
            entities = naive_extract_graph(query.split("\n")[0], self.nlp)["nouns"]
            shortest_path_k = kwargs.get("shortest_path_k", 4)
            min_count = kwargs.get("min_count", 2)
            if len(entities) == 0:
                # NOTE: without global retrieval, the result is empty.
                return {"chunks":""}
            wasd_res = self.wasd_step(entities, shortest_path_k, min_count)
            chunk_count = self._count_chunks(wasd_res)
            chunk_counts_history = []
            chunk_counts_history.append((shortest_path_k, min_count, chunk_count))

            if chunk_count == 0:
                query_embed = self.embedder.encode(query).reshape(1, -1) # need (1, -1) for faiss.
                _, condidate_chunks_indexs = self.faiss_index.search(query_embed, k = 25 *2)
                # the normal faiss index return the (1, k) shape. squeeze it to (k,).
                condidate_chunks_indexs = condidate_chunks_indexs[0]
                condidate_chunk_ids = [self.collapse_tree_ids[i] for i in condidate_chunks_indexs]
                filtered_chunk_ids = self._faiss_entity_filter(condidate_chunk_ids, entities) 
                # return the entity_entityB: [chunk_id1, chunk_id2, ...]
                res_str = self.format_res(filtered_chunk_ids)

                result = {"chunks":res_str}
                if kwargs.get("debug", True):
                    supplement_info = self._build_supplement_info(filtered_chunk_ids, entities, filtered_chunk_ids, list(filtered_chunk_ids.keys()), chunk_count, chunk_counts_history)
                    result.update(supplement_info)
                return result
            
            else:
                flag = 0
                while chunk_count > kwargs.get("max_chunk_setting", 25):
                    prev_wasd_res = copy.deepcopy(wasd_res)
                    if flag == 0:
                        shortest_path_k -= 1
                        flag = 1
                    else:
                        min_count += 1
                        flag = 0
                    wasd_res = self.wasd_step(entities, shortest_path_k, min_count)
                    chunk_count = self._count_chunks(wasd_res)
                    chunk_counts_history.append((shortest_path_k, min_count, chunk_count))
                if chunk_count != 0:
                    print("BOTTOM2TOP: final chunk_count", chunk_count)
                    res_str = self.format_res(wasd_res)

                    result = {"chunks":res_str}
                    if kwargs.get("debug", True):
                        supplement_info = self._build_supplement_info(wasd_res, entities, wasd_res, list(wasd_res.keys()), chunk_count, chunk_counts_history)
                        result.update(supplement_info)
                    return result
                else:
                    candidate_chunks = prev_wasd_res
                    res_ids = self.entityaware_filter(candidate_chunks, entities)
                    chunk_count = self._count_chunks(res_ids)
                    res_str = self.format_res(res_ids)
                    result = {"chunks":res_str}
                    result["chunk_counts_history"] = chunk_counts_history
                    if kwargs.get("debug", True):
                        supplement_info = self._build_supplement_info(res_ids, entities, res_ids, list(res_ids.keys()), chunk_count, chunk_counts_history)
                        result.update(supplement_info)
                    return result  


        elif ablation_type == "wo_entityawarefilter":
            entities = naive_extract_graph(query.split("\n")[0], self.nlp)["nouns"]
            shortest_path_k = kwargs.get("shortest_path_k", 4)
            min_count = kwargs.get("min_count", 2)
            if len(entities) == 0:
                chunk_ids = self.dense_retrieval(query, kwargs.get("max_chunk_setting", 25))
                result = {"chunks":self.format_res(chunk_ids)}
                if kwargs.get("debug", True):
                    supplement_info = self._build_supplement_info(chunk_ids, entities, chunk_ids, list(chunk_ids.keys()), len(chunk_ids), [])
                    result.update(supplement_info)
                    result["is_global_retrieval"] = True
                return result
            wasd_res = self.wasd_step(entities, shortest_path_k, min_count)
            chunk_count = self._count_chunks(wasd_res)
            chunk_counts_history = []
            chunk_counts_history.append((shortest_path_k, min_count, chunk_count))

            if chunk_count == 0:
                query_embed = self.embedder.encode(query).reshape(1, -1) # need (1, -1) for faiss.
                _, condidate_chunks_indexs = self.faiss_index.search(query_embed, k = 25 *2)
                # the normal faiss index return the (1, k) shape. squeeze it to (k,).
                condidate_chunks_indexs = condidate_chunks_indexs[0]
                condidate_chunk_ids = [self.collapse_tree_ids[i] for i in condidate_chunks_indexs]
                filtered_chunk_ids = self._faiss_entity_filter(condidate_chunk_ids, entities) 
                # return the entity_entityB: [chunk_id1, chunk_id2, ...]
                res_str = self.format_res(filtered_chunk_ids)

                result = {"chunks":res_str}
                if kwargs.get("debug", True):
                    supplement_info = self._build_supplement_info(filtered_chunk_ids, entities, filtered_chunk_ids, list(filtered_chunk_ids.keys()), 25, chunk_counts_history)
                    result.update(supplement_info)
                return result
            
            else:
                flag = 0
                while chunk_count > kwargs.get("max_chunk_setting", 25):
                    prev_wasd_res = copy.deepcopy(wasd_res)
                    if flag == 0:
                        shortest_path_k -= 1
                        flag = 1
                    else:
                        min_count += 1
                        flag = 0
                    wasd_res = self.wasd_step(entities, shortest_path_k, min_count)
                    chunk_count = self._count_chunks(wasd_res)
                    chunk_counts_history.append((shortest_path_k, min_count, chunk_count))
                if chunk_count != 0:
                    print("BOTTOM2TOP: final chunk_count", chunk_count)
                    res_str = self.format_res(wasd_res)

                    result = {"chunks":res_str}
                    if kwargs.get("debug", True):
                        supplement_info = self._build_supplement_info(wasd_res, entities, wasd_res, list(wasd_res.keys()), chunk_count, chunk_counts_history)
                        result.update(supplement_info)
                    return result
                else:
                    candidate_chunks = prev_wasd_res
                    all_chunks = []
                    for key in candidate_chunks:
                        all_chunks.extend(candidate_chunks[key])
                    if len(all_chunks) > 25:
                        selected_chunks = random.sample(all_chunks, 25)
                    else:
                        selected_chunks = all_chunks
                    
                    res_ids = {}
                    for chunk in selected_chunks:
                        # 找到chunk所属的key
                        for key in candidate_chunks:
                            if chunk in candidate_chunks[key]:
                                if key not in res_ids:
                                    res_ids[key] = []
                                res_ids[key].append(chunk)
                                break
                    chunk_count = self._count_chunks(res_ids)
                    res_str = self.format_res(res_ids)
                    result = {"chunks":res_str}
                    result["chunk_counts_history"] = chunk_counts_history
                    if kwargs.get("debug", True):
                        supplement_info = self._build_supplement_info(res_ids, entities, res_ids, list(res_ids.keys()), chunk_count, chunk_counts_history)
                        result.update(supplement_info)
                    return result  


        elif ablation_type == "wo_occurrencefilter":
            entities = naive_extract_graph(query.split("\n")[0], self.nlp)["nouns"]
            shortest_path_k = kwargs.get("shortest_path_k", 4)
            min_count = kwargs.get("min_count", 2)
            if len(entities) == 0:
                chunk_ids = self.dense_retrieval(query, kwargs.get("max_chunk_setting", 25))
                result = {"chunks":self.format_res(chunk_ids)}
                if kwargs.get("debug", True):
                    supplement_info = self._build_supplement_info(chunk_ids, entities, chunk_ids, list(chunk_ids.keys()), len(chunk_ids), [])
                    result.update(supplement_info)
                    result["is_global_retrieval"] = True
                return result
            wasd_res = self.wasd_step(entities, shortest_path_k, min_count)
            chunk_count = self._count_chunks(wasd_res)
            chunk_counts_history = []
            chunk_counts_history.append((shortest_path_k, min_count, chunk_count))

            if chunk_count == 0:
                # use dense retrieval to get the chunks.
                result = {"chunks":""}
                return result
            
            else:
                flag = 0
                while chunk_count > kwargs.get("max_chunk_setting", 25):
                    prev_wasd_res = copy.deepcopy(wasd_res)
                    if flag == 0:
                        shortest_path_k -= 1
                        flag = 1
                    else:
                        min_count += 1
                        flag = 0
                    wasd_res = self.wasd_step(entities, shortest_path_k, min_count)
                    chunk_count = self._count_chunks(wasd_res)
                    chunk_counts_history.append((shortest_path_k, min_count, chunk_count))
                if chunk_count != 0:
                    print("BOTTOM2TOP: final chunk_count", chunk_count)
                    res_str = self.format_res(wasd_res)

                    result = {"chunks":res_str}
                    if kwargs.get("debug", True):
                        supplement_info = self._build_supplement_info(wasd_res, entities, wasd_res, list(wasd_res.keys()), chunk_count, chunk_counts_history)
                        result.update(supplement_info)
                    return result
                else:
                    candidate_chunks = prev_wasd_res
                    res_ids = self.entityaware_filter(candidate_chunks, entities)
                    chunk_count = self._count_chunks(res_ids)
                    res_str = self.format_res(res_ids)
                    result = {"chunks":res_str}
                    result["chunk_counts_history"] = chunk_counts_history
                    if kwargs.get("debug", True):
                        supplement_info = self._build_supplement_info(res_ids, entities, res_ids, list(res_ids.keys()), chunk_count, chunk_counts_history)
                        result.update(supplement_info)
                    return result  


        elif ablation_type == "wo_GF&EAR":
            entities = naive_extract_graph(query.split("\n")[0], self.nlp)["nouns"]
            shortest_path_k = kwargs.get("shortest_path_k", 4)
            min_count = kwargs.get("min_count", 2)
            if len(entities) == 0:
                chunk_ids = self.dense_retrieval(query, kwargs.get("max_chunk_setting", 25))
                result = {"chunks":self.format_res(chunk_ids)}
                if kwargs.get("debug", True):
                    supplement_info = self._build_supplement_info(chunk_ids, entities, chunk_ids, list(chunk_ids.keys()), len(chunk_ids), [])
                    result.update(supplement_info)
                    result["is_global_retrieval"] = True
                return result
            wasd_res = self.wasd_ablation(entities, shortest_path_k, min_count, ablation_type)
            chunk_count = self._count_chunks(wasd_res)
            chunk_counts_history = []
            chunk_counts_history.append((shortest_path_k, min_count, chunk_count))

            if chunk_count == 0:
                query_embed = self.embedder.encode(query).reshape(1, -1) # need (1, -1) for faiss.
                _, condidate_chunks_indexs = self.faiss_index.search(query_embed, k = 25 *2)
                # the normal faiss index return the (1, k) shape. squeeze it to (k,).
                condidate_chunks_indexs = condidate_chunks_indexs[0]
                condidate_chunk_ids = [self.collapse_tree_ids[i] for i in condidate_chunks_indexs]
                filtered_chunk_ids = self._faiss_entity_filter(condidate_chunk_ids, entities) 
                # return the entity_entityB: [chunk_id1, chunk_id2, ...]
                res_str = self.format_res(filtered_chunk_ids)

                result = {"chunks":res_str}
                if kwargs.get("debug", True):
                    supplement_info = self._build_supplement_info(filtered_chunk_ids, entities, filtered_chunk_ids, list(filtered_chunk_ids.keys()), chunk_count, chunk_counts_history)
                    result.update(supplement_info)
                return result
            
            else:
                candidate_chunks = wasd_res
                all_chunks = []
                for key in candidate_chunks:
                    all_chunks.extend(candidate_chunks[key])
                if len(all_chunks) > 25:
                    selected_chunks = random.sample(all_chunks, 25)
                else:
                    selected_chunks = all_chunks
                
                res_ids = {}
                for chunk in selected_chunks:
                    # 找到chunk所属的key
                    for key in candidate_chunks:
                        if chunk in candidate_chunks[key]:
                            if key not in res_ids:
                                res_ids[key] = []
                            res_ids[key].append(chunk)
                            break
                chunk_count = self._count_chunks(res_ids)
                res_str = self.format_res(res_ids)
                result = {"chunks":res_str}
                result["chunk_counts_history"] = chunk_counts_history
                if kwargs.get("debug", True):
                    supplement_info = self._build_supplement_info(res_ids, entities, res_ids, list(res_ids.keys()), chunk_count, chunk_counts_history)
                    result.update(supplement_info)
                return result


        else:
            raise ValueError("ablation_type is not valid.")



    def query(self, query, **kwargs):
        if kwargs.get("ablation_study", False):
            # for ablation study.
            return self._ablation_query(query, **kwargs)

        # step 1: extract the Entities from the query.
        entities = naive_extract_graph(query.split("\n")[0], self.nlp)#
        entities = entities["nouns"]

        # step 2.0: set up the parameters.
        shortest_path_k = kwargs.get("shortest_path_k", 4)
        min_count = kwargs.get("min_count", 2)

        # step 2.1: short circuit, if there is no entity, then return the naive dense retrieval.
        if len(entities) == 0:
            chunk_ids = self.dense_retrieval(query, kwargs.get("max_chunk_setting", 25))
            result = {"chunks":self.format_res(chunk_ids)}
            if kwargs.get("debug", True):
                supplement_info = self._build_supplement_info(chunk_ids, entities, chunk_ids, list(chunk_ids.keys()), len(chunk_ids), [])
                result.update(supplement_info)
                result["retrieval_type"] = "Global Search"
            return result

        # step 2.2: initialize the chunks by wasd method.
        wasd_res = self.wasd_step(entities, shortest_path_k, min_count)

        # step 2.2: check the result.
            # if the chunk count is larger than the max chunk setting, then change the setting, increase the min count and decrease the shortest path k, until the chunk count is less than the max chunk setting.
            # if the chunk count is 0, take it as dense retrieval.
    
        chunk_count = self._count_chunks(wasd_res)
        # record the chunk count history, for debug.
        chunk_counts_history = []
        chunk_counts_history.append((shortest_path_k, min_count, chunk_count))

        # if the chunk count is 0, dense retrieval + entity filter.
        if chunk_count == 0:          
            query_embed = self.embedder.encode(query).reshape(1, -1) # need (1, -1) for faiss.
            _, condidate_chunks_indexs = self.faiss_index.search(query_embed, k = 25 *2)
            # the normal faiss index return the (1, k) shape. squeeze it to (k,).
            condidate_chunks_indexs = condidate_chunks_indexs[0]
            condidate_chunk_ids = [self.collapse_tree_ids[i] for i in condidate_chunks_indexs]
            filtered_chunk_ids = self._faiss_entity_filter(condidate_chunk_ids, entities) 
            # return the entity_entityB: [chunk_id1, chunk_id2, ...]
            res_str = self.format_res(filtered_chunk_ids)

            result = {"chunks":res_str}
            if kwargs.get("debug", True):
                supplement_info = self._build_supplement_info(filtered_chunk_ids, entities, filtered_chunk_ids, list(filtered_chunk_ids.keys()), 25, chunk_counts_history)
                result.update(supplement_info)
                result["retrieval_type"] = "Occurrence Rerank"
            return result
        

        while chunk_count > kwargs.get("max_chunk_setting", 25):
            prev_wasd_res = copy.deepcopy(wasd_res)
            # if the chunk count is larger than the max chunk setting
            # then change the setting, increase the min count and decrease the shortest path k.
            shortest_path_k -= 1
            # update the result with new restrictions.
            wasd_res = self.wasd_step(entities, shortest_path_k, min_count)
            chunk_count = self._count_chunks(wasd_res)
            chunk_counts_history.append((shortest_path_k, min_count, chunk_count))

        # if the chunk count is 0, dense retrieval + entity filter.
        # if the chunk count is not 0, return the result.
        if chunk_count != 0:
            # format the result.
            print("BOTTOM2TOP: final chunk_count", chunk_count)
            res_str = self.format_res(wasd_res)

            result = {"chunks":res_str}
            if kwargs.get("debug", True):
                supplement_info = self._build_supplement_info(wasd_res, entities, wasd_res, list(wasd_res.keys()), chunk_count, chunk_counts_history)
                result.update(supplement_info)
                result["retrieval_type"] = f"Local, Loop for {len(chunk_counts_history)-1} times"
            return result
        else:
            # the previous wasd result is not empty, so we can use it as candidate chunks.
            candidate_chunks = prev_wasd_res
            res_ids = self.entityaware_filter(candidate_chunks, entities)
            chunk_count = self._count_chunks(res_ids)
            res_str = self.format_res(res_ids)
            result = {"chunks":res_str}
            result["chunk_counts_history"] = chunk_counts_history
            if kwargs.get("debug", True):
                supplement_info = self._build_supplement_info(res_ids, entities, res_ids, list(res_ids.keys()), 25, chunk_counts_history)
                result.update(supplement_info)
                result["retrieval_type"] = f"EntityAware Filter, Loop for {len(chunk_counts_history)-1} times"
            return result        

    def _build_supplement_info(self, chunk_ids, entities, neighbor_nodes, keys, len_chunks, chunk_counts_history):
        return {
            "chunk_ids": chunk_ids,
            "entities": entities,
            "neighbor_nodes": neighbor_nodes,
            "keys": keys,
            "len_chunks": len_chunks,
            "chunk_counts_history": chunk_counts_history
        }
    

if __name__ == "__main__":
    import json
    from extract_graph import load_cache
    cache_tree = json.load(open("cache/wo_faiss/NovelQA/81/tree.json", "r"))
    G, index, appearance_count = load_cache("cache/wo_faiss/NovelQA/81")
    nlp = spacy.load("en_core_web_sm")
    retriever = Retriever(cache_tree, G, index, appearance_count, nlp, device="cuda:1")
    query = ["who commit suicide?","What is the main character of the novel?","Did Violet's heart become stone eventually?","Why did Molly want to find Longcoat Bob so eagerly?"]
    for q in query:
        result = retriever.query(q)
        print(result)