import os
import json
import argparse
import logging
from tqdm import tqdm
from typing import List
from utils import load_LLM, load_bert, load_sentence_bert
from prompts import NOVEL_TRIPLETS_PROMPT
from yb_dataloader import NovelQALoader

HF_TOKEN = "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"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


def extract_chunk(model, tokenizer, chunk):
    input_with_prompt = NOVEL_TRIPLETS_PROMPT.format(text = chunk)
    inputs = tokenizer(input_with_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_with_prompt):]

def triplets_cleaner(triplets:List[str]) -> List[str]:
    # remove empty and make it lowercase.
    triplets = [triplet.strip(")").strip("(").lower() for triplet in triplets if triplet.strip()]
    # extract the nodes and relations.
    nodes = set()
    relations = set()
    for triplet in triplets:
        if triplet.count(",") != 2:
            continue
        triplet = triplet.split(",")
        nodes.add(triplet[0].strip())
        nodes.add(triplet[2].strip())
        relations.add(triplet[1].strip())
    # remove duplicates
    triplets = list(set(triplets))
    nodes = list(nodes)
    relations = list(relations)
    return triplets, nodes, relations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()
    model, tokenizer = load_LLM(args.model_name)
    model.to(args.device)
    model.eval()

    logger.info("Loading data...")
    data = NovelQALoader(docpath="./NovelQA/Books", qapath="./NovelQA/Data",tokenizer=tokenizer,chunk_size=args.chunk_size,overlap=args.overlap)
    
    for i, one_book in tqdm(enumerate(data), total=len(data)):
        book_chunks = one_book["book_chunks"]
        Graph = []
        node_chunk_map = {}
        for j, chunk in tqdm(enumerate(book_chunks),total=len(book_chunks), desc=f"Book {i+1} of {len(data)}"):
            # logger.info("chunk: {}".format(chunk))
            triplets = extract_chunk(model, tokenizer, chunk)
            # logger.info("triplets generated: {}".format(triplets))
            triplets, nodes, relations = triplets_cleaner(triplets.split("\n"))
            logger.info("triplets count: {}".format(len(triplets)))
            dict_graph = {
                "nodes": nodes,
                "relations": relations,
                "triplets": triplets,
            }
            Graph.append(dict_graph)
            for node in nodes:
                if node not in node_chunk_map:
                    node_chunk_map[node] = []
                # the ith book and the jth chunk.
                node_chunk_map[node].append(j)

        # once the book is done, save the graph and node chunk map.
        with open(os.path.join(args.output_folder, f"{one_book['book_id']}.json"), "w") as f:
            json.dump(Graph, f)
        with open(os.path.join(args.output_folder, f"{one_book['book_id']}_node_chunk_map.json"), "w") as f:
            json.dump(node_chunk_map, f)

if __name__ == "__main__":
    main()