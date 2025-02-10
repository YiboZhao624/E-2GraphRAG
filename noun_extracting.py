import spacy
from yb_dataloader import NarrativeQALoader, NovelQALoader, chunk_index
from graphutils import merge_entities
from transformers import AutoTokenizer
import argparse
import logging
from collections import defaultdict
from tqdm import tqdm

logger = logging.getLogger("summarize")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def extract_nouns(text):
    try:
        nlp = spacy.load('en_core_web_lg')
    except OSError:
        logger.info("Downloading spacy model...")
        spacy.cli.download('en_core_web_lg')
        nlp = spacy.load('en_core_web_lg')
    
    # Process the text
    doc = nlp(text)
    
    # Store the noun and its cooccurrence information
    noun_pairs = {}
    all_nouns = set()
    double_nouns = {}  # Store two-word names mapping

    # Process each sentence
    for sent in doc.sents:
        sentence_terms = []
        
        ent_positions = set()
        for ent in sent.ents:
            if ent.label_ == 'PERSON':
                # Handle several-word person names
                name_parts = ent.text.split()
                if len(name_parts) >= 2:
                    for name in name_parts:
                        double_nouns[name] = name_parts
                    sentence_terms.extend(name_parts)
                else:
                    sentence_terms.append(ent.text)
            elif ent.label_ in ['ORG', 'GPE']:
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
        "cooccurrence": noun_pairs,
        "double_nouns": double_nouns
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="novelqa",choices = ["narrativeqa", "novelqa"])
    parser.add_argument("--model_name", type=str, default="/root/shared_planing/LLM_model/Qwen2.5-14B-Instruct")
    parser.add_argument("--resume", type=int, default=0)

    args = parser.parse_args()
    logger.info(args.dataset)
    logger.info(args.model_name)
    resume = args.resume
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.dataset == "narrativeqa":
        loader = NarrativeQALoader(saving_folder="./NarrativeQA", tokenizer=tokenizer, chunk_size=1200,overlap=100)
    elif args.dataset == "novelqa":
        loader = NovelQALoader(saving_folder="./NovelQA", tokenizer=tokenizer, chunk_size=1200,overlap=100)

    for i, data in enumerate(tqdm(loader, desc="Extracting Nouns", ncols=100)):
        if i < resume:
            continue
        book_id = data["book_id"]
        book = data["book"]
        book_chunks = data["book_chunks"]
        qa = data["qa"]
        book_dict : chunk_index = {
            "global_nouns": set(),
            "chunk_to_nouns": {},
            "noun_to_chunks": defaultdict(set),
            "noun_pairs": {},
            "double_nouns": {}
        }
        for chunk in book_chunks:
            chunk_text = chunk["text"]
            chunk_id = chunk["id"]
            nouns_info = extract_nouns(chunk_text)
            nouns = nouns_info["nouns"]
            cooccurrence = nouns_info["cooccurrence"]
            double_nouns = nouns_info["double_nouns"]
            
            # Update double_nouns dictionary
            book_dict["double_nouns"].update(double_nouns)
            
            # Check if any single name part exists and add its pair
            updated_nouns = set(nouns)
            for name in updated_nouns:
                to_update = double_nouns.get(name, [])
                if len(to_update) > 0:
                    updated_nouns.update(set(to_update))
            
            book_dict["global_nouns"].update(updated_nouns)
            book_dict["chunk_to_nouns"][chunk_id] = updated_nouns
            for noun in updated_nouns:
                book_dict["noun_to_chunks"][noun].add(chunk_id)
            for pair, count in cooccurrence.items():
                book_dict["noun_pairs"][pair] = book_dict["noun_pairs"].get(pair, 0) + count
        book_dict["noun_pairs"] = [(pair[0], pair[1], count) for pair, count in book_dict["noun_pairs"].items()]
        book_dict["noun_pairs"], book_dict["node_name_mapping"] = merge_entities(book_dict["noun_pairs"])
        loader.update_index(book_id, book_dict)
        loader.save_index(book_id, f"{loader.parent_folder}/New_Index")

if __name__ == "__main__":
    main()