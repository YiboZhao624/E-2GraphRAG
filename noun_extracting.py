import spacy
from yb_dataloader import NarrativeQALoader, NovelQALoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import logging

logger = logging.getLogger("summarize")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def extract_nouns(text):
    # Load English language model
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
        sentence_nouns = [token.text.lower() for token in sent if token.pos_ == "NOUN"]
        all_nouns.update(sentence_nouns)
        
        # Count the cooccurrence of nouns
        for i in range(len(sentence_nouns)):
            for j in range(i+1, len(sentence_nouns)):
                noun1, noun2 = sorted([sentence_nouns[i], sentence_nouns[j]])
                pair = (noun1, noun2)
                noun_pairs[pair] = noun_pairs.get(pair, 0) + 1
    
    return list(all_nouns), noun_pairs

def get_noun_cooccurrence(text):
    """get the cooccurrence of nouns in the text."""
    nouns, cooccurrence = extract_nouns(text)
    
    # sort the cooccurrence by the frequency.
    sorted_pairs = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "nouns": nouns,
        "cooccurrence": sorted_pairs
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="narrativeqa",choices = ["narrativeqa", "novelqa"])
    parser.add_argument("--model_name", type=str, default="")

    args = parser.parse_args()
    logger.info(args.dataset)
    logger.info(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.dataset == "narrativeqa":
        loader = NarrativeQALoader()
    elif args.dataset == "novelqa":
        loader = NovelQALoader(docpath="./NovelQA/Books", qapath="./NovelQA/Data", tokenizer=tokenizer, chunk_size=1200,overlap=100)

    for data in loader:
        book_id = data["book_id"]
        book = data["book"]
        book_chunks = data["book_chunks"]
        qa = data["qa"]
        for chunk in book_chunks:
            chunk_text = chunk["text"]
            nouns, cooccurrence = extract_nouns(chunk_text)
            logger.info(nouns)
            logger.info(cooccurrence)

            break
        break


if __name__ == "__main__":
    main()