import sentence_transformers
from tqdm import tqdm
import logging
import os
import faiss
import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)



class TreeRetriever:
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
    def __init__(self, args, dataloader):
        self.model_name = args.embedder_name
        self.model = sentence_transformers.SentenceTransformer(self.model_name)
        self.cache = args.embed_cache
        self.data = dataloader
        if not self.cache_dir:
            self.cache_dir = "./cache"
            logger.warning("Cache directory not specified, using default: ./cache")
        self.cache_dir = args.embed_cache_dir

        self.reset()

    def process_data(self):
        books_data = []
        for book in self.data:
            book_id = book['id']
            chunks = []
            for chunk in book["chunks"]:
                chunks.append(chunk)
            books_data.append({
                "id": book_id,
                "chunks": chunks
            })
        return books_data

    def reset(self):
        '''
        Reset the retriever by embedding the documents and storing the embeddings in the cache.
        if the cache is not empty, load the embeddings from the cache.
        else, create the embeddings and store them in the cache.
        TODO:
            - add the summary of the book to the cache.
        '''
        books_data = self.process_data()
        save_model_name = self.model_name.split('/')[-1]

        for book in books_data:
            book_id = book['id']
            # read the cache book by book.
            if self.cache and os.path.isfile(os.path.join(self.cache_dir, f'cache-{save_model_name}_{book_id}.pkl')):
                embeds, self.embeds_lookup = pickle.load(open(os.path.join(self.cache_dir, f'cache-{save_model_name}_{book_id}.pkl'), 'rb'))
                assert self.embeds_lookup == book_id
            # no cache, create the cache.
            else:
                embeds = self._infer(book['chunks'])
                self.embeds_lookup = book_id
                pickle.dump([embeds, book_id], open(os.path.join(self.cache_dir, f'cache-{save_model_name}_{book_id}.pkl'), 'wb'))

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
        
    @classmethod
    def build_embeddings(cls, model, corpus_dataset, args):
        retriever = cls(model, corpus_dataset, args)
        retriever.doc_embedding_inference()
        return retriever
    
    def query(self, query, book_id, located_chunks_id, args):
        # 1. find the father node of the chunks.
        # 2. calculate the embedding of the query.
        # 3. calculate the similarity between the query and the chunks.
        # 4. return the chunks in list ordered by similarity.
        
        pass