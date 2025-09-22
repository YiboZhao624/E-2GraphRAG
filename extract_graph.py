import os
import torch
from typing import List
import json, re
import spacy, nltk
import networkx as nx
from itertools import combinations
from typing import List, Tuple, Literal
import time
import logging
import threading
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Get logger for this module
logger = logging.getLogger(__name__)

def load_nlp(language:str="en", method: Literal["Spacy", "NLTK","BERT_NER_POS"]="Spacy"):
    if method == "Spacy":
        nlp = SpacyExtractor(language)
    elif method == "NLTK":
        nlp = NLTKExtractor(language)
    elif method == "BERT_NER_POS":
        nlp = BERTExtractor(language)
    return nlp
        
class Extractor:
    def __init__(self, language):
        self.language = language
        self.nlp = self.load_model(language)
        self.method = "Extractor"
    
    def load_model(self):
        raise NotImplementedError("Subclass must implement the load_model method.")

    def __call__(self, text:str):
        raise NotImplementedError("Subclass must implement __call__ method")
    
    def naive_extract_graph(self, text:str):
        raise NotImplementedError("Subclass must implement the naive_extract_graph method.")

class SpacyExtractor(Extractor):
    def __init__(self, language:str="en"):
        super().__init__(language)
        self.nlp = self.load_model(language)
        self.method = "Spacy"
    
    def load_model(self, language):
        if language == "en":
            try:
                nlp = spacy.load("en_core_web_lg")
            except:
                logger.info("Downloading spacy model...")
                spacy.cli.download("en_core_web_lg")
                nlp = spacy.load("en_core_web_lg")
        elif language == "zh":
            try:
                nlp = spacy.load("en_core_web_lg")
            except:
                logger.info("Downloading spacy model...")
                spacy.cli.download("en_core_web_lg")
                nlp = spacy.load("en_core_web_lg")
        return nlp
    
    def naive_extract_graph(self, text: str):
        doc = self.nlp(text)

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
    
class NLTKExtractor(Extractor):
    _nltk_initialized = False
    _nltk_init_lock = threading.Lock()
    def __init__(self, language:str="en"):
        super().__init__(language)
        self.nlp = self.load_model(language)
        self.method = "NLTK"

    def load_model(self, language):
        """
        The core logic that performs the one-time, thread-safe initialization.
        This method contains your original code, adapted for this pattern.
        """
        # 1. Fast, lock-free check. If already initialized, do nothing.
        if NLTKExtractor._nltk_initialized:
            return
        # 2. If not initialized, acquire lock to prevent race conditions
        with NLTKExtractor._nltk_init_lock:
            # 3. Double-check after acquiring the lock, in case another thread finished
            #    while this one was waiting.
            if NLTKExtractor._nltk_initialized:
                return
            
            logger.info("="*10)
            logger.info("First-time setup: Running thread-safe NLTK initialization...")
            logger.info("="*10)

            data_dir = "/root/nltk_data"
            if not os.path.exists(data_dir):
                logger.info(f"NLTK dir does not exist, now creating.")
                os.makedirs(data_dir)
            if data_dir not in nltk.data.path:
                logger.info(f"Adding '{data_dir}' to NLTK search path.")
                nltk.data.path.append(data_dir)
            else:
                logger.info(f"NLTK data directory '{data_dir}' is already in the search path.")
            required_packages = {
                'tokenizers/punkt': 'punkt',
                'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
                'chunkers/maxent_ne_chunker': 'maxent_ne_chunker',
                'corpora/words': 'words',
                'taggers/averaged_perceptron_tagger_eng': 'averaged_perceptron_tagger_eng',
                'chunkers/maxent_ne_chunker_tab': 'maxent_ne_chunker_tab'
            }
            all_packages_available = True

            for resource_path, package_id in required_packages.items():
                try:
                    nltk.data.find(resource_path)
                except LookupError:
                    all_packages_available = False
                    logger.info(f"Package {package_id} is missing, now downloading...")
                    nltk.download(package_id, download_dir=data_dir)
                    logger.info(f"Package {package_id} downloaded.")

            if all_packages_available:
                logger.info("All required NLTK packages are ready.")
            else:
                logger.info("Some packages are missing, now downloading...")
            NLTKExtractor._nltk_initialized = True
            return None

    def naive_extract_graph(self, text: str):
        sentences = nltk.tokenize.sent_tokenize(text)

        # noun pairs provide the edge.
        noun_pairs = {}

        # all_nouns saving the nodes.
        all_nouns = set()

        # process the name like John Brown
        double_nouns = {}
        appearance_count = {}

        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            tagged_tokens = nltk.pos_tag(tokens)
            
            # Extract named entities using NLTK's NER
            ne_tree = nltk.ne_chunk(tagged_tokens)
            
            sentence_terms = []
            ent_positions = set()
            
            # Process named entities
            for chunk in ne_tree:
                if hasattr(chunk, 'label'):
                    if chunk.label() == 'PERSON':
                        # handle the name like John Brown, John Brown Smith.
                        name_parts = [word for word, pos in chunk.leaves()]
                        if len(name_parts) >= 2:
                            for name in name_parts:
                                double_nouns[name] = name_parts
                            sentence_terms.extend(name_parts)
                            for name in name_parts:
                                appearance_count[name] = appearance_count.get(name, 0) + 1
                        else:
                            sentence_terms.append(' '.join(name_parts))
                            appearance_count[' '.join(name_parts)] = appearance_count.get(' '.join(name_parts), 0) + 1
                    
                    # process the organization or country.
                    elif chunk.label() in ["ORGANIZATION", "GPE"]:
                        entity_text = ' '.join([word for word, pos in chunk.leaves()])
                        sentence_terms.append(entity_text)
                        appearance_count[entity_text] = appearance_count.get(entity_text, 0) + 1
                    
                    # Mark entity positions to avoid double counting
                    for word, pos in chunk.leaves():
                        ent_positions.add(word)
            
            # Process regular nouns and proper nouns
            for word, pos in tagged_tokens:
                if word in ent_positions:
                    continue
                if pos.startswith('NN') and word.strip():
                    # Convert to lowercase for common nouns, keep proper nouns as is
                    if pos == 'NN' or pos == 'NNS':
                        sentence_terms.append(word.lower())
                        appearance_count[word.lower()] = appearance_count.get(word.lower(), 0) + 1
                    elif pos == 'NNP' or pos == 'NNPS':
                        sentence_terms.append(word)
                        appearance_count[word] = appearance_count.get(word, 0) + 1
            
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

class BERTExtractor(Extractor):
    """
    使用BERT模型进行命名实体识别（NER）和词性标注（POS）以提取名词的提取器。
    """
    def __init__(self, language: str = "en", ner_model_name = "./models/ner", pos_model_name = "./models/pos"):
        """
        初始化BERTExtractor。

        Args:
            language (str): 语言, 当前实现主要支持 'en'。
        """
        self.ner_model_name = ner_model_name
        self.pos_model_name = pos_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.ner_model_name)
        super().__init__(language)
        self.nlp_pipelines = self.load_model(language)
        self.method = "BERT_NER_POS"

    def load_model(self, language):
        """
        load the BERT model and tokenizer for NER and POS.

        Returns:
            A dictionary containing the NER and POS transformer pipelines.
        """
        if language != "en":
            logger.warning(f"The current BERT models are primarily for English. Performance may vary.")

        try:
            device = 0 if torch.cuda.is_available() else -1
            
            # NER pipeline
            ner_pipeline = pipeline(
                "ner",
                model=self.ner_model_name,
                tokenizer=self.ner_model_name,
                device=device,
                grouped_entities=True
            )
            logger.info(f"BERT NER model '{self.ner_model_name}' loaded successfully.")
            
            # POS pipeline
            pos_pipeline = pipeline(
                "token-classification",
                model=self.pos_model_name,
                tokenizer=self.pos_model_name,
                device=device,
                aggregation_strategy="simple" # Groups sub-tokens (e.g., 'engineer', '##ing')
            )
            logger.info(f"BERT POS model '{self.pos_model_name}' loaded successfully.")

            return {"ner": ner_pipeline, "pos": pos_pipeline}
        except Exception as e:
            logger.error(f"Failed to load BERT models: {e}")
            raise

    def __call__(self, text: str):
        """
        use the BERT model to process the text to extract entities and nouns.
        """
        return self.naive_extract_graph(text)

    def naive_extract_graph(self, text: str):
        """
        extract the entities and nouns (as the nodes of the graph) and their cooccurrence relations (as the edges of the graph) from the text.
        because the bert model only process the text within 512 tokens, we split the chunks into smaller sub-chunks and then aggregate the results.
        """
        # aggregate the results.
        all_terms = set()
        appearance_count = {}

        # --- 新增的文本切分逻辑 ---
        max_length = 480  # 设置一个保守的长度，给特殊符号留出空间
        overlap = 50      # 设置重叠大小，以保持上下文连续性

        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs['input_ids'][0]

        sub_chunks = []
        start = 0
        # 使用滑动窗口切分input_ids
        while start < len(input_ids):
            end = start + max_length
            sub_chunk_ids = input_ids[start:end]
            # 将切分后的token ids解码回文本
            sub_chunk_text = self.tokenizer.decode(sub_chunk_ids, skip_special_tokens=True)
            sub_chunks.append(sub_chunk_text)
            if end >= len(input_ids):
                break
            start += max_length - overlap

        # --- 对每个子块进行分析并合并结果 ---
        for sub_chunk in sub_chunks:
            ner_results = self.nlp['ner'](sub_chunk)
            pos_results = self.nlp['pos'](sub_chunk)

            for entity in ner_results:
                all_terms.add(entity['word'].strip())

            for token in pos_results:
                if token['entity_group'] in ['NOUN', 'PROPN']:
                    all_terms.add(token['word'].strip())

        noun_pairs = {}
        sentences = nltk.sent_tokenize(text) # 使用原始文本进行句子分割

        for sentence in sentences:
            sentence_terms = set()
            for term in all_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', sentence, re.IGNORECASE):
                    sentence_terms.add(term)
                    appearance_count[term] = appearance_count.get(term, 0) + 1
            
            # Count the cooccurrence of terms
            if len(sentence_terms) > 1:
                for pair in combinations(sorted(list(sentence_terms)), 2):
                    key = tuple(pair)
                    noun_pairs[key] = noun_pairs.get(key, 0) + 1

        return {
            "nouns": list(all_terms),
            "cooccurrence": noun_pairs,
            "double_nouns": {},
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

def load_cache(cache_path:str, method_name:str):
    graph_file_path = os.path.join(cache_path, f"graph_{method_name}.json")
    index_file_path = os.path.join(cache_path, f"index_{method_name}.json")
    appearance_count_file_path = os.path.join(cache_path, f"appearance_count_{method_name}.json")
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
    
def extract_graph(text:List[str], cache_folder:str, nlp:Extractor, use_cache=True, reextract=False):
    extract_start_time = time.time()
    if use_cache and os.path.exists(os.path.join(cache_folder, f"graph_{nlp.method}.json")) and os.path.exists(os.path.join(cache_folder, f"index_{nlp.method}.json")) and os.path.exists(os.path.join(cache_folder, f"appearance_count_{nlp.method}.json")):
        method_name = nlp.method
        return load_cache(cache_folder, method_name), -1
    else:
        graph_file_path = os.path.join(cache_folder, f"graph_{nlp.method}.json")
        index_file_path = os.path.join(cache_folder, f"index_{nlp.method}.json")
        appearance_count_file_path = os.path.join(cache_folder, f"appearance_count_{nlp.method}.json")
        edges = []
        index = {}
        appearance_count = {}

        for i, chunk in enumerate(text):
            if i % 10 == 1:
                logger.info(f"Now extracting the {i}th chunk...")
            naive_result = nlp.naive_extract_graph(chunk)
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
        # save the graph and index.
        save_graph(edges, graph_file_path)
        save_index(index, index_file_path)
        save_appearance_count(appearance_count, appearance_count_file_path)
        extract_end_time = time.time()
        return (G, index, appearance_count), extract_end_time - extract_start_time

if __name__ == '__main__':
    bert_extractor = BERTExtractor()

    sample_text = (
        "John Doe, a software engineer at Google, visited New York last week. "
        "He met with Jane Smith from Microsoft to discuss a potential partnership. "
        "The meeting took place in the Empire State Building."
    )

    graph_info = bert_extractor(sample_text)

    import json
    print("Extracted Information:")
    print(graph_info)

    print("\nNouns (Entities and Common Nouns):")
    print(sorted(graph_info['nouns']))

    print("\nCo-occurrence Pairs:")
    for pair, weight in graph_info['cooccurrence'].items():
        print(f"- {pair}: {weight} time(s)")