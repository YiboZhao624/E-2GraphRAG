from sentence_transformers import SentenceTransformer
from yb_dataloader import AbstractDataLoader
from tqdm import tqdm
import numpy as np
import logging
import os
import faiss
import pickle
from utils import extract_nouns
from graphutils import multi_shortest_path


logger = logging.getLogger("TAGRetriever")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)



class TAGRetriever:
    '''
    First, using the args and data to initialize the retriever.
    The data is a dataloader defined in the `dataloader.py`.
    During the initialization, the retriever will
        1. check if the cache is built. if not, build the cache.
        2. load the embeddings from the cache.
        3. initialize the index and add the embeddings to the index using faiss.
    To use the TreeRetriever, call the query method.
    For query method, you should input the query and the number of chunks you have located.
    The query method will first find the father node of the chunks and then filter the chunks by the query. Finally, it will return the chunks in list ordered by similarity.
    '''
    def __init__(self, dataloader:AbstractDataLoader, book_id, embedder_model, embed_model_name, device, embedding_cache_path = None):
        self.model = embedder_model
        self.model.to(device)
        self.model.eval()
        self.embed_model_name = embed_model_name
        self.data = dataloader.dataset[book_id]
        self.graph = dataloader.graph[book_id]
        # self.graph has already been a networkx graph. it is processed in the dataloader.
        self.tree_structure = dataloader.tree_structure[book_id]
        self._index = dataloader._index[book_id]
        self.book_id = book_id
        self.cache_dir = embedding_cache_path
        if not embedding_cache_path:
            self.cache_dir = f"./cache/{self.book_id}"
            logger.warning(f"Cache directory not specified, using default: {self.cache_dir}")
        self.id_to_index = {}
        self.chunk_id_to_chunk = {
            chunk["id"]: chunk 
            for chunk in self.data["book_chunks"]
        }
        self.chunk_id_to_chunk.update(
            {
                summary["id"]: summary
                for depth, summaries in self.data["summary_layers"].items()
                for summary in summaries
            }
        )
        self.index_to_id = {}

        self.reset()

    def reset(self):
        '''
        Reset the retriever by embedding the documents and storing the embeddings in the cache.
        if the cache is not empty, load the embeddings from the cache.
        else, create the embeddings and store them in the cache.
        '''
        save_model_name = self.embed_model_name.split('/')[-1]
        cache_file = os.path.join(self.cache_dir, f'cache-{save_model_name}_{self.book_id}.pkl')

        if self.cache_dir and os.path.isfile(cache_file):
            embeds, self.id_to_index, self.index_to_id = pickle.load(open(cache_file, "rb"))
        else:
            text_to_encode = []
            chunk_ids = []

            logger.info("Loading chunks")
            for chunk in self.data["book_chunks"]:
                text_to_encode.append(chunk["text"])
                chunk_ids.append(chunk["id"])

            if "summary_layers" in self.data:
                logger.info("Loading summaries")
                for depth, summaries in self.data["summary_layers"].items():
                    for summary in summaries:
                        text_to_encode.append(summary["text"])
                        chunk_ids.append(summary["id"])

            embeds = self._infer(text_to_encode)
            self.id_to_index = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
            logger.info("Embeddings inferred")
            self.index_to_id = {i: chunk_id for chunk_id, i in self.id_to_index.items()}
            if self.cache_dir:
                logger.info("Saving embeddings to cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                pickle.dump([embeds, self.id_to_index, self.index_to_id], open(cache_file, "wb"))

        self.init_index_and_add(embeds)

    def _infer(self, docs):
        '''
        Infer the embeddings for the chunks within a book.
        '''
        logger.info("Inferring embeddings")
        embeds = self.model.encode(docs,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeds
    
    def _initialize_faiss_index(self, dim: int):
        '''
        Initialize a cpu index with fixed dimensions.
        '''
        logger.info("Initializing FAISS index")
        self.index = None
        cpu_index = faiss.IndexFlatIP(dim)
        self.index = cpu_index

    def _move_index_to_gpu(self):
        logger.info("Moving index to GPU")
        ngpu = faiss.get_num_gpus()
        gpu_resources = []
        for i in range(ngpu):
            res = faiss.StandardGpuResources()
            gpu_resources.append(res)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        self.index = faiss.index_cpu_to_gpu_multiple(vres, vdev, self.index, co)

    def init_index_and_add(self, embeds):
        '''
        Initialize the index and add the embeddings to the index.
        using the above methods.
        '''
        logger.info("Initialize the index...")
        dim = embeds.shape[1]
        self._initialize_faiss_index(dim)
        self.index.add(embeds)
        
        # if self.use_gpu:
        #     self._move_index_to_gpu()
        #     logger.info("Index moved to GPU")
        logger.info("Index initialized and embeddings added")

    def query_subset(self, query_embed, node_ids, k):
        '''
        Query the subset of the index.
        '''
        # 获取所有有效索引
        # logger.info(f"length of node_ids: {len(node_ids)}")
        # logger.info(f"node_ids: {node_ids}")
        valid_indices = [self.id_to_index[node_id] for node_id in node_ids]
        # logger.info(f"length of valid_indices: {len(valid_indices)}")
        # logger.info(f"valid_indices: {valid_indices}")
        valid_indices = np.array(valid_indices)
        # logger.info(f"length of total indices: {self.index.ntotal}")
        # 对整个索引进行搜索
        distances, indices = self.index.search(query_embed.reshape(1,-1), len(valid_indices))
        distances = distances[0]
        indices = indices[0]
        
        # 过滤出只在 valid_indices 中的结果
        filtered_results = []
        filtered_distances = []
        for d, idx in zip(distances, indices):
            if idx in valid_indices:
                filtered_results.append(idx)
                filtered_distances.append(d)
                if len(filtered_results) == k:
                    break
        
        # 将索引转换回原始的 node_ids
        result_ids = []
        for idx in filtered_results:
            for node_id, index in self.id_to_index.items():
                if index == idx:
                    result_ids.append(node_id)
                    break
                
        return np.array(filtered_distances), result_ids
    
    def graph_query(self, entities):
        # find the shortest path between the entities.
        # return the path in List[tuple(entity, entity, count)].
        # count is the number of cooccurrences between the two entities.
        paths = multi_shortest_path(self.graph, entities)
        return paths
    
    def find_chunks(self, paths):
        # 1. simply find the chunks related to the nodes in the path.
        res_chunks = set()
        for path in paths:
            res_chunk = set()
            if len(path) <= 2:
                for node in path:
                    if node in self._index["global_nouns"]:
                        res_chunk.update(self._index["noun_to_chunks"][node])
            if len(path) > 2:
                # A -> B -> C
                # (A ∩ B) ∪ (B ∩ C)
                for i in range(len(path) - 1):
                    node1, node2 = path[i], path[i+1]
                    if node1 in self._index["global_nouns"]:
                        chunks1 = self._index["noun_to_chunks"][node1]
                    else:
                        # logger.info(f"Node: {node1} not in global_nouns")
                        chunks1 = set()
                    if node2 in self._index["global_nouns"]:
                        chunks2 = self._index["noun_to_chunks"][node2]
                    else:
                        # logger.info(f"Node: {node2} not in global_nouns")
                        chunks2 = set()
                    res_chunk.update(chunks1 | chunks2)
            res_chunks.update(res_chunk)
        res_chunks = list(res_chunks)
        return res_chunks
    

    def query_all(self, query_embed, k):
        distances, indices = self.index.search(query_embed.reshape(1,-1), k)
        return distances, indices
    
    def query(self, query, book_id):
        # 1. NER the query, find the entities in the query.
        # 2. find the shortest path between the entities.
        # 3. calculate the embedding of the query.
        # 4. using the entities inside the path to find the chunks in the book.
        # 5. calculate the similarity between the query and the chunks.
        # 6. return the chunks in list ordered by similarity.
        question = query.split("\n")[0]
        entities:list = extract_nouns(question)["nouns"]
        logger.info(f"Query text: {question}")
        logger.info(f"Extracted entities: {entities}")
        # step 2.
        paths = self.graph_query(entities)
        related_entities = set()
        for path in paths:
            for node in path:
                related_entities.add(node)
        related_entities = list(related_entities)
        # logger.info(f"Related entities: {related_entities}")
        # step 3.
        query_embed = self.model.encode(question, convert_to_numpy=True)
        # step 4.
        # multiple paths/ long path -> (A ∩ B)∪(B ∩ C) if a——>b——>c
        chunk_ids = self.find_chunks(paths)
        # logger.info(f"Chunk ids count: {len(chunk_ids)}")
        # step 5. subset query.
        if len(chunk_ids) == 0:
            distances, indices = self.query_all(query_embed, 10)
            distances = distances[0]
            indices = indices[0]
            # print("distances", distances)
            # print("indices", indices)
            # print("index_to_id", self.index_to_id)
            # print("id_to_index", self.id_to_index)
            chunk_ids = [self.index_to_id[idx] for idx in indices]
            res_chunks = [self.chunk_id_to_chunk[chunk_id] for chunk_id in chunk_ids]
        else:
            # logger.info(f"Query subset with chunk ids: {chunk_ids}")
            distances, indices = self.query_subset(query_embed, chunk_ids, 10)
            # logger.info(f"length of indices: {len(indices)}")
            res_chunks = [self.chunk_id_to_chunk[chunk_id] for chunk_id in indices]
        logger.info(f"length of res_chunks: {len(res_chunks)}")
        # 按相似度排序
        chunk_id_to_distance = {chunk_id: distances[i] for i, chunk_id in enumerate(indices)}
        # res_chunks = sorted(res_chunks, key=lambda x: chunk_id_to_distance[x["id"]], reverse=True)

        return res_chunks, related_entities, entities
