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
        self.book_id = book_id
        self.cache_dir = embedding_cache_path
        if not embedding_cache_path:
            self.cache_dir = f"./cache/{self.book_id}"
            logger.warning(f"Cache directory not specified, using default: {self.cache_dir}")
        self.id_to_index = {}

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
            embeds, self.id_to_index = pickle.load(open(cache_file, "rb"))
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
            if self.cache_dir:
                logger.info("Saving embeddings to cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                pickle.dump([embeds, self.id_to_index], open(cache_file, "wb"))

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
        valid_indices = [self.id_to_index[node_id] for node_id in node_ids]
        valid_indices = np.array(valid_indices)

        id_selector = faiss.IDSelectorBatch(len(valid_indices), faiss.swig_ptr(valid_indices))

        distances, indices = self.index.search_with_selector(
            query_embed.reshape(1,-1),
            k,
            id_selector
        )
        result_ids = []
        for idx in indices[0]:
            for node_id, index in self.id_to_index.items():
                if index == idx:
                    result_ids.append(node_id)
                    break
                    
        return distances[0], result_ids        
    
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
                    if node in self.tree_structure[self.book_id]["nodes"]:
                        res_chunk.add(self.tree_structure[self.book_id]["nodes"][node])
            if len(path) > 2:
                # A -> B -> C
                # (A ∩ B) ∪ (B ∩ C)
                for i in range(len(path) - 1):
                    node1, node2 = path[i], path[i+1]

                    chunks1 = set()
                    chunks2 = set()
                    for node in node1:
                        if node in self.tree_structure[self.book_id]["nodes"]:
                            chunks1.add(self.tree_structure[self.book_id]["nodes"][node])
                    for node in node2:
                        if node in self.tree_structure[self.book_id]["nodes"]:
                            chunks2.add(self.tree_structure[self.book_id]["nodes"][node])
                    res_chunk.update(chunks1 & chunks2)
            res_chunks.update(res_chunk)
        res_chunks = [chunk["id"] for chunk in res_chunks]
        return res_chunks
    
    def query(self, query, book_id):
        # 1. NER the query, find the entities in the query.
        # 2. find the shortest path between the entities.
        # 3. calculate the embedding of the query.
        # 4. using the entities inside the path to find the chunks in the book.
        # 5. calculate the similarity between the query and the chunks.
        # 6. return the chunks in list ordered by similarity.
        question = query.split("\n")[0]
        entities:list = extract_nouns(question)
        # step 2.
        paths = self.graph_query(entities)
        related_entities = set()
        for path in paths:
            for node in path:
                related_entities.add(node)
        related_entities = list(related_entities)
        # step 3.
        query_embed = self.model.encode(question, convert_to_numpy=True)
        # step 4.
        # multiple paths/ long path -> (A ∩ B)∪(B ∩ C) if a——>b——>c
        chunk_ids = self.find_chunks(paths)
        # step 5. subset query.
        distances, indices = self.query_subset(query_embed, chunk_ids, 10)
        # step 6. return the chunks in list ordered by similarity.
        res_chunks = [self.data["book_chunks"][idx] for idx in indices]
        res_chunks = sorted(res_chunks, key=lambda x: distances[indices.index(self.id_to_index[x["id"]])])

        return res_chunks, related_entities
