from sentence_transformers import SentenceTransformer
from yb_dataloader import AbstractDataLoader
from tqdm import tqdm
import numpy as np
import logging
import os
import faiss
import pickle
from utils import extract_nouns


logger = logging.getLogger(__name__)
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
    def __init__(self, dataloader:AbstractDataLoader, book_id, embedder_name, device, embedding_cache_path = None):
        self.model_name = embedder_name
        self.model = SentenceTransformer(self.model_name)
        self.model.to(device)
        self.model.eval()
        self.data = dataloader.dataset[book_id]
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
        TODO:
            - add the summary of the book to the cache.
        '''
        save_model_name = self.model_name.split('/')[-1]
        cache_file = os.path.join(self.cache_dir, f'cache-{save_model_name}_{self.book_id}.pkl')

        if self.cache and os.path.isfile(cache_file):
            embeds, self.id_to_index = pickle.load(open(cache_file, "rb"))
        else:
            text_to_encode = []
            chunk_ids = []

            for chunk in self.data["book_chunks"]:
                text_to_encode.append(chunk["text"])
                chunk_ids.append(chunk["id"])

            if "summary_layers" in self.data:
                for depth, summaries in self.data["summary_layers"].items():
                    for summary in summaries:
                        text_to_encode.append(summary["text"])
                        chunk_ids.append(summary["id"])

            embeds = self._infer(text_to_encode)
            self.id_to_index = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}

            if self.cache:
                os.makedirs(self.cache_dir, exist_ok=True)
                pickle.dump([embeds, self.id_to_index], open(cache_file, "wb"))

        self.init_index_and_add(embeds)

    def _infer(self, docs):
        '''
        Infer the embeddings for the chunks within a book.
        '''
        embeds = self.model.encode(docs,
            batch_size=4,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeds
    
    def _initialize_faiss_index(self, dim: int):
        '''
        Initialize a cpu index with fixed dimensions.
        '''
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
        
        if self.use_gpu:
            self._move_index_to_gpu()
            logger.info("Index moved to GPU")
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
    
    def graph_query(self, entities, book_id):
        # find the shortest path between the entities.
        # return the path in List[tuple(entity, entity, count)].
        # count is the number of cooccurrences between the two entities.
        pass
    
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
        path = self.graph_query(entities, book_id)
        # step 3.
        query_embed = self.model.encode(question, convert_to_numpy=True)
        # step 4.


    @classmethod
    def build_embeddings(cls, model, corpus_dataset, args):
        retriever = cls(model, corpus_dataset, args)
        retriever.doc_embedding_inference()
        return retriever