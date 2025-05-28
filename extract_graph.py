import os
from typing import List
import json
import pickle
import spacy
import networkx as nx
import torch
from itertools import combinations
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cluster import DBSCAN
import time

def load_nlp(language:str="en"):
    if language == "en":
        try:
            nlp = spacy.load("en_core_web_lg")
        except:
            print("Downloading spacy model...")
            spacy.cli.download("en_core_web_lg")
            nlp = spacy.load("en_core_web_lg")
    elif language == "zh":
        try:
            nlp = spacy.load("zh_core_web_lg")
        except:
            print("Downloading spacy model...")
            spacy.cli.download("zh_core_web_lg")
            nlp = spacy.load("zh_core_web_lg")
    return nlp
        
def naive_extract_graph(text:str, nlp:spacy.Language):
    # process the text
    doc = nlp(text)
    
    # noun pairs provide the edge.
    noun_pairs = {}

    # all_nouns saving the nodes.
    all_nouns = set()

    # process the name like John Brown
    double_nouns = {}
    appearance_count = {}

    # TODO: 一个chunk里的连通还是一个句子里连通？
    for sent in doc.sents:
        sentence_terms = []

        ent_positions = set()
        for ent in sent.ents:
            if ent.label_ == "PERSON":
                # handle the name like John Brown, John Brown Smith.
                name_parts = ent.text.split()
                if len(name_parts) >= 2:
                    for name in name_parts:
                        double_nouns[name] = name_parts
                    sentence_terms.extend(name_parts)
                    for name in name_parts:
                        appearance_count[name] = appearance_count.get(name, 0) + 1
                else:
                    sentence_terms.append(ent.text)
                    appearance_count[ent.text] = appearance_count.get(ent.text, 0) + 1
            
            # process the organization or country.
            elif ent.label_ in ["ORG", "GPE"]:
                sentence_terms.append(ent.text)
                appearance_count[ent.text] = appearance_count.get(ent.text, 0) + 1
            for token in ent:
                ent_positions.add(token.i)

        for token in sent:
            if token.i in ent_positions:
                continue
            if token.pos_ == "NOUN" and token.lemma_.strip():
                sentence_terms.append(token.lemma_.lower())
                appearance_count[token.lemma_.lower()] = appearance_count.get(token.lemma_.lower(), 0) + 1
            elif token.pos_ == "PROPN" and token.text.strip():
                sentence_terms.append(token.lemma_.lower())
                appearance_count[token.lemma_.lower()] = appearance_count.get(token.lemma_.lower(), 0) + 1
            elif token.pos_ == "PROPN" and token.text.strip():
                sentence_terms.append(token.text)
                appearance_count[token.text] = appearance_count.get(token.text, 0) + 1
                
        all_nouns.update(sentence_terms)
        
        # Count the cooccurrence of terms
        for i in range(len(sentence_terms)):
            for j in range(i+1, len(sentence_terms)):
                term1, term2 = sorted([sentence_terms[i], sentence_terms[j]])
                pair = (term1, term2)
                noun_pairs[pair] = noun_pairs.get(pair, 0) + 1
    
    return {
        "nouns": list(all_nouns),
        "cooccurrence": noun_pairs,
        "double_nouns": double_nouns,
        "appearance_count": appearance_count
    }


def build_graph(triplets: List[Tuple[str, str, int]]) -> nx.Graph:
    '''
    build the graph from the triplets, merging weights of duplicate edges
    Args:
        triplets: List of [node1, node2, weight] List
    Returns:
        NetworkX graph with merged weights
    '''
    G = nx.Graph()
    
    # 创建字典来存储边的权重和
    edge_weights = {}
    for n1, n2, weight in triplets:
        # 因为是无向图，所以(a,b)和(b,a)是相同的边
        edge = tuple(sorted([n1, n2]))
        edge_weights[edge] = edge_weights.get(edge, 0) + weight
    
    # 将合并后的边添加到图中
    for (n1, n2), weight in edge_weights.items():
        G.add_edge(n1, n2, weight=weight)
    
    return G

def get_shortest_path(G:nx.Graph, start:str, end:str) -> List[str]:
    '''
    get the shortest path between start and end.
    '''
    try:
        return nx.shortest_path(G, start, end, weight='weight')
    except nx.NodeNotFound:
        print(f"NodeNotFound: {start} or {end}")
        return []

def multi_shortest_path(G:nx.Graph, entities:List[str]) -> List[List[str]]:
    '''
    get the shortest path between all entities.
    '''
    paths = []
    for i, j in combinations(entities, 2):
        path = get_shortest_path(G, i, j)
        paths.append(path)
    return paths

# not used.
def merge_entities(nouns: List[str], eps: float = 0.15, min_samples: int = 2, debug: bool = True) -> dict:
    '''
    merge the entities into the graph using sklearn clustering
    Args:
        nouns: List of noun entities
        eps: similarity threshold
        min_samples: minimum samples for clustering
        debug: whether to print debug information
    '''
    import transformers
    model = transformers.AutoModel.from_pretrained("bert-base-cased").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    nodes = list(set(nouns))
    
    # 获取词向量
    node_embeddings = []
    batch_size = 32
    for i in range(0, len(nodes), batch_size):
        batch_nodes = nodes[i:i + batch_size]
        tokenized = tokenizer(batch_nodes, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokenized)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        node_embeddings.extend(batch_embeddings.tolist())
    
    # 转换为numpy数组
    node_embeddings = np.array(node_embeddings)
    
    # 计算并打印相似度矩阵
    if debug:
        similarity_matrix = cosine_similarity(node_embeddings)
        print("\n词语相似度矩阵:")
        # 打印表头
        print(f"{'':15}", end='')
        for node in nodes:
            print(f"{node:>10}", end='')
        print("\n" + "-" * (15 + 10 * len(nodes)))
        
        # 打印相似度值
        for i, node1 in enumerate(nodes):
            print(f"{node1:15}", end='')
            for j, node2 in enumerate(nodes):
                sim = similarity_matrix[i][j]
                # 高亮显示高相似度对
                if i != j and sim > 1 - eps:
                    print(f"\033[92m{sim:10.3f}\033[0m", end='')  # 绿色显示
                else:
                    print(f"{sim:10.3f}", end='')
            print()
        
        # 打印高相似度对
        print("\n相似度高于阈值的词对 (sim > {:.2f}):".format(1- eps))
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                sim = similarity_matrix[i][j]
                if sim > 1 - eps:
                    print(f"{nodes[i]:15} - {nodes[j]:15} = {sim:.3f}")
    
    # 原有的聚类逻辑
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = clustering.fit_predict(node_embeddings)
    
    # 打印聚类结果
    if debug:
        print("\n聚类结果:")
        clusters = {}
        for node, label in zip(nodes, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node)
        for label, members in clusters.items():
            print(f"Cluster {label}: {members}")
    
    # 构建映射
    node_name_mapping = {}
    for node, label in zip(nodes, cluster_labels):
        if label == -1:
            node_name_mapping[node] = node
        else:
            cluster_members = [n for n, l in zip(nodes, cluster_labels) if l == label]
            rep = max(cluster_members, key=len)
            node_name_mapping[node] = rep
            
    return node_name_mapping

def load_cache(cache_path:str):
    graph_file_path = os.path.join(cache_path, "graph.json")
    index_file_path = os.path.join(cache_path, "index.json")
    appearance_count_file_path = os.path.join(cache_path, "appearance_count.json")
    edges = json.load(open(graph_file_path, "r"))
    index = json.load(open(index_file_path, "r"))
    appearance_count = json.load(open(appearance_count_file_path, "r"))
    graph = build_graph(edges)
    return graph, index, appearance_count

def save_graph(result, cache_path:str):
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=4)

def save_index(result, cache_path:str):
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=4)

def save_appearance_count(result, cache_path:str):
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=4)
    
def extract_graph(text:List[str], cache_folder:str, nlp:spacy.Language, use_cache=True, reextract=False):
    extract_start_time = time.time()
    if use_cache and os.path.exists(os.path.join(cache_folder, "graph.json")) and os.path.exists(os.path.join(cache_folder, "index.json")) and os.path.exists(os.path.join(cache_folder, "appearance_count.json")):
        return load_cache(cache_folder), -1
    else:
        graph_file_path = os.path.join(cache_folder, "graph.json")
        index_file_path = os.path.join(cache_folder, "index.json")
        appearance_count_file_path = os.path.join(cache_folder, "appearance_count.json")
        edges = []
        index = {}
        appearance_count = {}

        for i, chunk in enumerate(text):
            naive_result = naive_extract_graph(chunk, nlp)
            # not merge the entities.
            appearance_count["leaf_{}".format(i)] = naive_result["appearance_count"]

            for noun in naive_result["nouns"]:
                if noun not in index:
                    index[noun] = []
                index[noun].append("leaf_{}".format(i))
            
            for noun, count in naive_result["appearance_count"].items():
                appearance_count[noun] = appearance_count.get(noun, 0) + count

            # add the cooccurrence.
            for pair, weight in naive_result["cooccurrence"].items():
                head, tail = pair
                edges.append([head, tail, weight])

        # build the graph.
        G = build_graph(edges)
        if reextract == True:
            save_appearance_count(appearance_count, appearance_count_file_path)
            return (G, index, appearance_count), -1
        # save the graph and index.
        save_graph(edges, graph_file_path)
        save_index(index, index_file_path)
        save_appearance_count(appearance_count, appearance_count_file_path)
        extract_end_time = time.time()
        return (G, index, appearance_count), extract_end_time - extract_start_time

if __name__ == "__main__":
    # nouns = [
    #     "desk","chair","table","book","computer","mouse",
    #     "keyboard","screen","printer","laptop","notebook","PC"
    # ]
    # print("原始词语:", nouns)
    # result = merge_entities(nouns, eps=0.08, min_samples=2, debug=True)
    # print("\n最终合并结果:")
    # for orig, merged in sorted(result.items()):
    #     if orig != merged:
    #         print(f"{orig} -> {merged}")

    # 测试用例
    edges = [
        ('a', 'b', 1),
        ('a', 'b', 3),
        ('b', 'a', 2)
    ]
    
    G = build_graph(edges)
    
    # 打印所有边的权重
    for u, v, w in G.edges(data='weight'):
        print(f"Edge ({u}, {v}): weight = {w}")  # 应该输出: Edge (a, b): weight = 6






def coref_extract_graph(text:str):
    try:
        nlp = spacy.load("en_core_web_lg")
    except:
        print("Downloading spacy model...")
        spacy.cli.download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")
    
    nlp.add_pipe("coreferee")
    doc = nlp(text)
    
    # resolve the coreference.
    chains = doc._.coref_chains

    # TODO: if it is necessary?
    pass

