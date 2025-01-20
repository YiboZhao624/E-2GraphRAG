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
        sentence_terms = []
        
        ent_positions = set()
        for ent in sent.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                sentence_terms.append(ent.text)
                for token in ent:
                    ent_positions.add(token.i)
        
        for token in sent:
            if token.i in ent_positions:
                continue
            if token.pos_ == "NOUN" and token.lemma_.strip():
                sentence_terms.append(token.lemma_.lower())
            elif token.pos_ == "PROPN" and token.text.strip():
                sentence_terms.append(token.text)
        
        all_nouns.update(sentence_terms)
        
        # Count the cooccurrence of terms
        for i in range(len(sentence_terms)):
            for j in range(i+1, len(sentence_terms)):
                term1, term2 = sorted([sentence_terms[i], sentence_terms[j]])
                pair = (term1, term2)
                noun_pairs[pair] = noun_pairs.get(pair, 0) + 1
    
    return {
        "nouns": list(all_nouns),
        "cooccurrence": noun_pairs
    }