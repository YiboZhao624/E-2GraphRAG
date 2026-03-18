import os
import json
import re
import time
import logging
import threading
from itertools import combinations
from typing import List, Tuple, Literal, Dict, Any, Optional
import yaml

import networkx as nx
from prompt_dict import Prompts
from llm_providers import create_llm


def _get_spacy():
    import spacy

    return spacy


def _get_spacy_cli():
    import spacy

    return spacy.cli


def _get_nltk():
    import nltk  # type: ignore

    return nltk


def _get_torch():
    import torch

    return torch


def _get_transformers_pipeline():
    from transformers import pipeline

    return pipeline


def _get_auto_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer


def _get_hanlp():
    import hanlp

    return hanlp

# Get logger for this module
logger = logging.getLogger(__name__)

def load_nlp(
    language: str = "en",
    method: Literal["Spacy", "NLTK", "BERT_NER_POS", "HanLP", "LLM", "LLMVerifier"] = "Spacy",
    **kwargs,
):
    if method == "Spacy":
        nlp = SpacyExtractor(language)
    elif method == "NLTK":
        nlp = NLTKExtractor(language)
    elif method == "BERT_NER_POS":
        nlp = BERTExtractor(
            language=language,
            ner_model_name=kwargs.get("ner_model_name", "./models/ner"),
            pos_model_name=kwargs.get("pos_model_name", "./models/pos"),
        )
    elif method == "HanLP":
        nlp = HanLPExtractor(
            language=language,
            ner_model_path=kwargs.get("ner_model_path", "./models/ner"),
            pos_model_path=kwargs.get("pos_model_path", "./models/pos"),
            hanlp_root=kwargs.get("hanlp_root", "./hanlp"),
            require_local=kwargs.get("hanlp_require_local", False),
        )
    elif method == "LLM":
        nlp = LLMExtractor(
            language=language,
            llm_config=kwargs.get("llm_config"),
            prompt_name=kwargs.get("prompt_name", "extract_entities_json"),
        )
    elif method == "LLMVerifier":
        nlp = LLMVerifier(
            language=language,
            llm_config=kwargs.get("llm_config"),
            graph_prompt_name=kwargs.get("graph_prompt_name", "extract_graph_relations_json"),
        )
    return nlp
        
class Extractor:
    def __init__(self, language):
        self.language = language
        self.nlp = self.load_model(language)
        self.method = "Extractor"
    
    def load_model(self):
        raise NotImplementedError("Subclass must implement the load_model method.")

    def __call__(self, text: str):
        # 默认调用 naive_extract_graph，子类可覆盖
        return self.naive_extract_graph(text)
    
    def naive_extract_graph(self, text:str):
        raise NotImplementedError("Subclass must implement the naive_extract_graph method.")

class SpacyExtractor(Extractor):
    def __init__(self, language:str="en"):
        super().__init__(language)
        self.nlp = self.load_model(language)
        self.method = "Spacy"
    
    def load_model(self, language):
        spacy = _get_spacy()
        spacy_cli = _get_spacy_cli()

        if language == "en":
            try:
                nlp = spacy.load("en_core_web_lg")
            except Exception:
                logger.info("Downloading spacy model...")
                spacy_cli.download("en_core_web_lg")
                nlp = spacy.load("en_core_web_lg")
        elif language == "zh":
            try:
                nlp = spacy.load("en_core_web_lg")
            except Exception:
                logger.info("Downloading spacy model...")
                spacy_cli.download("en_core_web_lg")
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
        nltk = _get_nltk()
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
        nltk = _get_nltk()
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
    def __init__(self, language: str = "en", ner_model_name="./models/ner", pos_model_name="./models/pos"):
        """
        初始化BERTExtractor。

        Args:
            language (str): 语言, 当前实现主要支持 'en'。
        """
        self.ner_model_name = ner_model_name
        self.pos_model_name = pos_model_name
        AutoTokenizer = _get_auto_tokenizer()
        self.tokenizer = AutoTokenizer.from_pretrained(self.ner_model_name)
        super().__init__(language)
        self.nlp_pipelines = self.nlp  # 兼容已有属性命名
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
            torch = _get_torch()
            transformers_pipeline = _get_transformers_pipeline()
            device = 0 if torch.cuda.is_available() else -1
            
            # NER pipeline
            ner_pipeline = transformers_pipeline(
                "ner",
                model=self.ner_model_name,
                tokenizer=self.ner_model_name,
                device=device,
                grouped_entities=True
            )
            logger.info(f"BERT NER model '{self.ner_model_name}' loaded successfully.")
            
            # POS pipeline
            pos_pipeline = transformers_pipeline(
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
        nltk = _get_nltk()
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


class HanLPExtractor(Extractor):
    """
    使用 HanLP 2.x 多任务模型（tok+pos+ner）进行中文实体/名词提取，避免 JVM 依赖。
    """

    def __init__(
        self,
        language: str = "zh",
        ner_model_path: str | None = None,
        pos_model_path: str | None = None,
        hanlp_root: str | None = None,
        require_local: bool | None = None,
        min_len: int = 1,
        min_freq: int = 1,
    ):
        # 参数保持兼容，但在 2.x 中不再需要 JVM 本地模型
        self.ner_model_path = ner_model_path
        self.pos_model_path = pos_model_path
        self.hanlp_root = hanlp_root
        self.require_local = require_local
        self.min_len = min_len
        self.min_freq = min_freq
        super().__init__(language)
        self.method = "HanLP"

    def load_model(self, language):
        # 默认使用 PyTorch 后端，避免 TensorFlow 兼容问题
        os.environ.setdefault("HANLP_BACKEND", "torch")

        hanlp = _get_hanlp()
        try:
            mtl = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
        except Exception as e:
            logger.error(f"Failed to load HanLP 2.x models: {e}")
            raise

        return {"mtl": mtl}

    def naive_extract_graph(self, text: str):
        """
        基于句子级共现统计实体/名词（HanLP 2.x 输出）。
        """
        noun_pairs = {}
        all_terms = set()
        appearance_count = {}
        double_nouns = {}

        # 句子划分（简单按中英文标点）
        sentences = [s.strip() for s in re.split(r"[。！？!?]", text) if s.strip()]

        def _ner_spans_to_texts(spans, sentence, tokens):
            """将 ner span 转文本，兼容多种格式；优先按字符起止索引截取原句。"""
            ents = []
            for sp in spans:
                if isinstance(sp, dict):
                    if "text" in sp:
                        ents.append(sp["text"])
                        continue
                    start = sp.get("start") if sp.get("start") is not None else sp.get("begin")
                    end = sp.get("end") if sp.get("end") is not None else sp.get("finish")
                elif isinstance(sp, (list, tuple)) and len(sp) >= 2:
                    # 可能是 (text, label) 或 (start, end, label)
                    if isinstance(sp[0], str) and not isinstance(sp[1], str):
                        ents.append(sp[0])
                        continue
                    start, end = sp[0], sp[1]
                else:
                    continue
                try:
                    start_i, end_i = int(start), int(end)
                    # HanLP mtl 的 ner/msra 返回字符级下标
                    if 0 <= start_i < end_i <= len(sentence):
                        ents.append(sentence[start_i:end_i])
                    elif 0 <= start_i < end_i <= len(tokens):
                        ents.append("".join(tokens[start_i:end_i]))
                except Exception:
                    continue
            return ents

        for sent in sentences:
            mtl_out = self.nlp["mtl"](sent, tasks=["tok", "pos", "ner"])
            # 兼容多种 key 命名（tok/tok/fine/tok/coarse、pos、ner/**）
            def _get_first_list(prefix: str):
                for k, v in mtl_out.items():
                    if k.startswith(prefix) and isinstance(v, list):
                        return v
                return []

            tokens = mtl_out.get("tok", []) or _get_first_list("tok")
            pos_tags = mtl_out.get("pos", []) or _get_first_list("pos")
            ner_spans = mtl_out.get("ner", []) or _get_first_list("ner")

            # 若返回是[[...]]这种嵌套，取第一层展开
            if tokens and isinstance(tokens[0], list):
                tokens = tokens[0]
            if pos_tags and isinstance(pos_tags[0], list):
                pos_tags = pos_tags[0]
            if ner_spans and isinstance(ner_spans[0], list):
                ner_spans = ner_spans[0]

            sentence_terms = []

            # 先收集 NER 实体
            for ent_text in _ner_spans_to_texts(ner_spans, sent, tokens):
                ent_text = ent_text.strip()
                if not ent_text:
                    continue
                sentence_terms.append(ent_text)
                appearance_count[ent_text] = appearance_count.get(ent_text, 0) + 1
                all_terms.add(ent_text)

            # 再收集名词/专有名词
            for tok, pos_tag in zip(tokens, pos_tags):
                if not isinstance(tok, str) or not isinstance(pos_tag, str):
                    continue
                tok_val = tok.strip()
                pos_val = pos_tag.strip()
                if not tok_val or not pos_val:
                    continue

                if pos_val.lower().startswith("n"):  # HanLP 2.x POS: NN/NR/NT... 大写
                    sentence_terms.append(tok_val)
                    appearance_count[tok_val] = appearance_count.get(tok_val, 0) + 1
                    all_terms.add(tok_val)

            # 去重后再统计共现，避免同词自环
            filtered_terms = [t for t in sentence_terms if len(t) >= self.min_len]
            unique_terms = list(set(filtered_terms))
            for i in range(len(unique_terms)):
                for j in range(i + 1, len(unique_terms)):
                    term1, term2 = sorted([unique_terms[i], unique_terms[j]])
                    if term1 == term2:
                        continue
                    pair = (term1, term2)
                    noun_pairs[pair] = noun_pairs.get(pair, 0) + 1

        kept_terms = {t for t, c in appearance_count.items() if len(t) >= self.min_len and c >= self.min_freq}
        filtered_pairs = {
            pair: w for pair, w in noun_pairs.items()
            if pair[0] in kept_terms and pair[1] in kept_terms
        }

        return {
            "nouns": list(kept_terms),
            "cooccurrence": filtered_pairs,
            "double_nouns": double_nouns,
            "appearance_count": {t: c for t, c in appearance_count.items() if t in kept_terms},
        }


class LLMExtractor(Extractor):
    """
    使用大模型直接抽取实体，要求模型按指定 JSON 模板输出实体列表。
    """

    def __init__(
        self,
        language: str = "zh",
        llm_config: Optional[Dict[str, Any]] = None,
        prompt_name: str = "extract_entities_json",
    ):
        self.llm_config = llm_config or {}
        self.prompt_name = prompt_name
        self.prompt_template = Prompts.get(prompt_name, "")
        if not self.prompt_template:
            raise ValueError(f"Prompt '{prompt_name}' not found in Prompts.")
        super().__init__(language)
        # 保持 self.llm 作为底层大模型实例，便于调用 generate
        self.llm = self.nlp
        self.method = "LLM"

    def load_model(self, language):
        if not self.llm_config:
            raise ValueError("llm_config is required to initialize LLMExtractor.")
        return create_llm(self.llm_config)

    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip().split("</think>")[-1].strip()
        if cleaned.startswith("```"):
            # 兼容 ```json 包裹的输出
            cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned)
            cleaned = cleaned.rsplit("```", 1)[0]
        try:
            return json.loads(cleaned)
        except Exception:
            try:
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(cleaned[start : end + 1])
            except Exception as e:
                logger.error(f"Failed to parse LLM JSON output: {e}, {cleaned}")
        return {}

    def _generate_and_parse(self, prompt: str, max_retries: int = 2) -> Dict[str, Any]:
        """LLM 调用带简单重试的解析包装，防止偶发输出格式错误。"""
        last_err = None
        for _ in range(max_retries + 1):
            try:
                response = self.llm.generate(prompt, max_new_tokens=1024*12).text
            except Exception as e:
                last_err = e
                logger.error(f"LLM generate failed: {e}")
                continue
            if not response:
                continue
            data = self._parse_response(response)
            if data:
                return data
        if last_err:
            logger.error(f"LLM generate/parse failed after retries: {last_err}")
        return {}

    def naive_extract_graph(self, text: str):
        prompt = self.prompt_template.format(content=text)
        data = self._generate_and_parse(prompt)
        entities_raw = data.get("entities") or data.get("nouns") or []
        nouns: List[str] = []
        for ent in entities_raw:
            if isinstance(ent, str):
                ent_name = ent.strip()
                if ent_name:
                    nouns.append(ent_name)
            elif isinstance(ent, dict):
                name = ent.get("name") or ent.get("entity") or ent.get("text")
                if isinstance(name, str) and name.strip():
                    nouns.append(name.strip())
        # 去重但保持顺序
        nouns = list(dict.fromkeys(nouns))

        appearance_count: Dict[str, int] = {}
        noun_pairs: Dict[Tuple[str, str], int] = {}

        sentences = [s.strip() for s in re.split(r"[。！？!?\.]", text) if s.strip()]
        for sent in sentences:
            sentence_terms = []
            for noun in nouns:
                if noun and noun in sent:
                    sentence_terms.append(noun)
                    appearance_count[noun] = appearance_count.get(noun, 0) + 1
            unique_terms = sorted(set(sentence_terms))
            for i in range(len(unique_terms)):
                for j in range(i + 1, len(unique_terms)):
                    pair = (unique_terms[i], unique_terms[j])
                    noun_pairs[pair] = noun_pairs.get(pair, 0) + 1

        return {
            "nouns": nouns,
            "cooccurrence": noun_pairs,
            "double_nouns": {},
            "appearance_count": appearance_count,
        }


class LLMVerifier(Extractor):
    """
    使用大模型一次性输出实体和关系，替代逐对验证。
    """

    def __init__(
        self,
        language: str = "zh",
        llm_config: Optional[Dict[str, Any]] = None,
        graph_prompt_name: str = "extract_graph_relations_json",
    ):
        self.llm_config = llm_config or {}
        self.graph_prompt_template = Prompts.get(graph_prompt_name, "")
        if not self.graph_prompt_template:
            raise ValueError(f"Prompt '{graph_prompt_name}' not found in Prompts.")
        super().__init__(language)
        self.llm = self.nlp
        self.method = "LLMVerifier"

    def load_model(self, language):
        if not self.llm_config:
            raise ValueError("llm_config is required to initialize LLMVerifier.")
        return create_llm(self.llm_config)

    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip().split("</think>")[-1].strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned)
            cleaned = cleaned.rsplit("```", 1)[0]
        try:
            return json.loads(cleaned)
        except Exception:
            try:
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(cleaned[start : end + 1])
            except Exception as e:
                logger.error(f"Failed to parse LLM JSON output: {e}, {cleaned}")
        return {}

    def _generate_and_parse(self, prompt: str, max_retries: int = 2) -> Dict[str, Any]:
        """LLM 调用带简单重试的解析包装，防止偶发输出格式错误。"""
        last_err = None
        for _ in range(max_retries + 1):
            try:
                response = self.llm.generate(prompt).text
            except Exception as e:
                last_err = e
                logger.error(f"LLM generate failed: {e}")
                continue
            if not response:
                continue
            data = self._parse_response(response)
            if data:
                return data
        if last_err:
            logger.error(f"LLM generate/parse failed after retries: {last_err}")
        return {}

    def naive_extract_graph(self, text: str):
        prompt = self.graph_prompt_template.format(content=text)
        data = self._generate_and_parse(prompt)

        entities_raw = data.get("entities") or data.get("nodes") or []
        relations_raw = data.get("relations") or data.get("edges") or []

        nouns: List[str] = []
        for ent in entities_raw:
            if isinstance(ent, str):
                name = ent.strip()
            elif isinstance(ent, dict):
                name = (ent.get("name") or ent.get("entity") or ent.get("text") or "").strip()
            else:
                name = ""
            if name:
                nouns.append(name)
        nouns = list(dict.fromkeys(nouns))

        appearance_count: Dict[str, int] = {}
        for noun in nouns:
            appearance_count[noun] = len(re.findall(re.escape(noun), text))

        noun_pairs: Dict[Tuple[str, str], int] = {}
        for rel in relations_raw:
            if isinstance(rel, dict):
                h = rel.get("head") or rel.get("source") or rel.get("from")
                t = rel.get("tail") or rel.get("target") or rel.get("to")
                w = rel.get("weight") or rel.get("count") or rel.get("confidence") or 1
            elif isinstance(rel, (list, tuple)) and len(rel) >= 2:
                h, t = rel[0], rel[1]
                w = rel[2] if len(rel) > 2 else 1
            else:
                continue
            if not isinstance(h, str) or not isinstance(t, str):
                continue
            h, t = h.strip(), t.strip()
            if not h or not t or h == t:
                continue
            pair = tuple(sorted([h, t]))
            noun_pairs[pair] = noun_pairs.get(pair, 0) + (w if isinstance(w, (int, float)) else 1)

        return {
            "nouns": nouns,
            "cooccurrence": noun_pairs,  # 表示有边，权重可累加
            "double_nouns": {},
            "appearance_count": appearance_count,
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
    """
    简易测试：使用 request_llm 配置（hanlp_request_llm_config_32b_rewrite.yaml）调用 LLMExtractor 抽取实体。
    依赖：本地推理服务需可用；需在配置中填好有效的 Authorization。
    """
    text = "客户分为几档"
    nlp = load_nlp(
        language="zh",
        method="HanLP",
    )
    result = nlp.naive_extract_graph(text)
    print(result)
    raise

    config_path = "./configs/hanlp_request_llm_config_32b_rewrite.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load config '{config_path}': {e}")
        raise

    llm_config = cfg.get("llm", {})
    extractor_cfg = cfg.get("extractor", {})
    language = extractor_cfg.get("language", "zh")

    sample_text = (
        "注册登录流程：客户打开APP判断是否已登录，未登录点击立即申请进入注册/登录；"
        "判断是否闪电贷预授信、是否实名、是否外部迁入、是否有额度等，分别进入对应流程。"
    )

    llm_extractor = load_nlp(
        language=language,
        method="LLM",
        llm_config=llm_config,
        prompt_name="extract_entities_json",
    )

    result = llm_extractor.naive_extract_graph(sample_text)
    print("LLM Extracted Information:")

    def _stringify_cooccurrence(co):
        return {f"{k[0]}__{k[1]}": v for k, v in co.items()}

    printable = dict(result)
    printable["cooccurrence"] = _stringify_cooccurrence(result.get("cooccurrence", {}))
    print(json.dumps(printable, ensure_ascii=False, indent=2))

    # --- LLMVerifier 测试：一次性实体+关系 ---
    llm_verifier = load_nlp(
        language=language,
        method="LLMVerifier",
        llm_config=llm_config,
        graph_prompt_name="extract_graph_relations_json",
    )

    verifier_result = llm_verifier.naive_extract_graph(sample_text)
    print("\nLLMVerifier Extracted Information:")
    printable_v = dict(verifier_result)
    printable_v["cooccurrence"] = _stringify_cooccurrence(verifier_result.get("cooccurrence", {}))
    print(json.dumps(printable_v, ensure_ascii=False, indent=2))