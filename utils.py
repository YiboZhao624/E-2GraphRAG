import json, os
import spacy


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def load_LLM(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_bert(model_name):
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_sentence_bert(model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model


def chunk_text(text, tokenizer_name, chunk_size, overlap):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i+chunk_size]
        chunks.append(chunk)
    return chunks


def extract_nouns(text):
    # Load English language model
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Downloading spacy model...")
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    
    # Process the text
    doc = nlp(text)
    
    # Store the noun and its cooccurrence information
    noun_pairs = {}
    all_nouns = set()
    
    # Process each sentence
    for sent in doc.sents:
        # Extract nouns from the sentence
        sentence_nouns = [token.lemma_.lower() for token in sent
                          if token.pos_ == "NOUN" and token.lemma_.strip()]
        all_nouns.update(sentence_nouns)
        
        # Count the cooccurrence of nouns
        for i in range(len(sentence_nouns)):
            for j in range(i+1, len(sentence_nouns)):
                noun1, noun2 = sorted([sentence_nouns[i], sentence_nouns[j]])
                pair = (noun1, noun2)
                noun_pairs[pair] = noun_pairs.get(pair, 0) + 1
    
    return {
        "nouns": list(all_nouns),
        "cooccurrence": noun_pairs
    }

def get_noun_cooccurrence(text):
    """get the cooccurrence of nouns in the text."""
    nouns, cooccurrence = extract_nouns(text)
    
    # sorted_pairs = sorted(cooccurrence.items(), 
    #                       key = lambda x: (-x[1], x[0][0], x[0][1]))
    pairs = list(cooccurrence.items())  # [((noun1, noun2), frequency), ...]
    sorted_pairs = sorted(pairs,
                         key=lambda pair: pair[1],
                         reverse=True)
    
    return {
        "nouns": nouns,
        "cooccurrence": sorted_pairs
    }