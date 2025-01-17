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
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        logger.info("Downloading spacy model...")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="novelqa",choices = ["narrativeqa", "novelqa"])
    parser.add_argument("--model_name", type=str, default="/root/shared_planing/LLM_model/Qwen2.5-14B-Instruct")

    args = parser.parse_args()
    logger.info(args.dataset)
    logger.info(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.dataset == "narrativeqa":
        loader = NarrativeQALoader(saving_folder="./NarrativeQA", tokenizer=tokenizer, chunk_size=1200,overlap=100)
    elif args.dataset == "novelqa":
        loader = NovelQALoader(saving_folder="./NovelQA", tokenizer=tokenizer, chunk_size=1200,overlap=100)

    for data in tqdm(loader, desc="Extracting Nouns", ncols=100):
        book_id = data["book_id"]
        book = data["book"]
        book_chunks = data["book_chunks"]
        qa = data["qa"]
        book_dict : chunk_index = {
            "global_nouns": set(),
            "chunk_to_nouns": {},
            "noun_to_chunks": defaultdict(set),
            "noun_pairs": {}
        }
        for chunk in book_chunks:
            chunk_text = chunk["text"]
            chunk_id = chunk["id"]
            nouns_info = extract_nouns(chunk_text)
            nouns = nouns_info["nouns"]
            cooccurrence = nouns_info["cooccurrence"]
            
            # logger.info(nouns)
            # logger.info(cooccurrence)
            book_dict["global_nouns"].update(nouns)
            book_dict["chunk_to_nouns"][chunk_id] = set(nouns)
            for noun in nouns:
                book_dict["noun_to_chunks"][noun].add(chunk_id)
            for pair, count in cooccurrence.items():
                book_dict["noun_pairs"][pair] = book_dict["noun_pairs"].get(pair, 0) + count
        book_dict["noun_pairs"] = [(pair[0], pair[1], count) for pair, count in book_dict["noun_pairs"].items()]
        book_dict["noun_pairs"], book_dict["node_name_mapping"] = merge_entities(book_dict["noun_pairs"])
        loader.update_index(book_id, book_dict)
        loader.save_index(book_id, f"{loader.parent_folder}/New_Index")

if __name__ == "__main__":
    main()