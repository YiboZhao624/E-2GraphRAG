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
    def __init__(self, args, data):
        self.model_name = args.embedder_name
        self.model = sentence_transformers.SentenceTransformer(self.model_name)
        self.cache = args.embed_cache
        self.data = data
        if not self.cache_dir:
            self.cache_dir = "./cache"
            logger.warning("Cache directory not specified, using default: ./cache")
        self.cache_dir = args.embed_cache_dir

        self.reset()

    def process_data(self):
        chunks = []
        ids = []
        meta_type = []

        for node_type_key in self.data.keys():
            node_type = node_type_key.split('_nodes')[0]
            logger.info(f'loading text for {node_type}')
            for nid in tqdm(self.data[node_type_key]):
                chunks.append(str(self.data[node_type_key][nid]['features'][self.node_text_keys[node_type][0]]))
                ids.append(nid)
                meta_type.append(node_type)
        return chunks, ids, meta_type

    def reset(self):
        '''
        Reset the retriever by embedding the documents and storing the embeddings in the cache.
        if the cache is not empty, load the embeddings from the cache.
        else, create the embeddings and store them in the cache.
        '''
        docs, ids, meta_type = self.process_graph()
        save_model_name = self.model_name.split('/')[-1]

        if self.cache and os.path.isfile(os.path.join(self.cache_dir, f'cache-{save_model_name}.pkl')):
            embeds, self.doc_lookup, self.doc_type = pickle.load(open(os.path.join(self.cache_dir, f'cache-{save_model_name}.pkl'), 'rb'))
            assert self.doc_lookup == ids
            assert self.doc_type == meta_type
        else:
            embeds = self._infer(docs)
            self.doc_lookup = ids
            self.doc_type = meta_type
            pickle.dump([embeds, ids, meta_type], open(os.path.join(self.cache_dir, f'cache-{save_model_name}.pkl'), 'wb'))

        self.init_index_and_add(embeds)

    def _infer(self, docs):
        '''
        Infer the embeddings for the documents.
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

    @classmethod
    def build_embeddings(cls, model, corpus_dataset, args):
        retriever = cls(model, corpus_dataset, args)
        retriever.doc_embedding_inference()
        return retriever