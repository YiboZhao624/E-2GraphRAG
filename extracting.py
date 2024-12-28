import os
import json
import argparse
from typing import List
from utils import load_LLM, load_bert, load_sentence_bert
from prompts import BIO_TRIPLETS_PROMPT

HF_TOKEN = "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"

def extract_chunk(model, tokenizer, chunk):
    input_with_prompt = BIO_TRIPLETS_PROMPT.format(chunk)
    inputs = tokenizer(input_with_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_with_prompt):]    

def triplets_cleaner(triplets:List[str]) -> List[str]:
    # remove empty and make it lowercase.
    triplets = [triplet.strip().lower() for triplet in triplets]
    # extract the nodes and relations.
    nodes = set()
    relations = set()
    for triplet in triplets:
        triplet = triplet.split(",")
        nodes.add(triplet[0])
        nodes.add(triplet[2])
        relations.add(triplet[1])
    # remove duplicates
    triplets = list(set(triplets))
    nodes = list(nodes)
    relations = list(relations)
    return triplets, nodes, relations

def load_data(data_path:str) -> List[str]:
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()
    model, tokenizer = load_LLM(args.model_name)
    model.to(args.device)
    model.eval()

    data = load_data(args.data_path)

    Graph = []
    node_chunk_map = {}
    for i, chunk in enumerate(data):
        triplets = extract_chunk(model, tokenizer, chunk)
        triplets, nodes, relations = triplets_cleaner(triplets)
        dict_graph = {
            "nodes": nodes,
            "relations": relations,
            "triplets": triplets,
        }
        for node in nodes:
            if node not in node_chunk_map:
                node_chunk_map[node] = []
            node_chunk_map[node].append(i)
        Graph.append(dict_graph)

    with open(os.path.join(args.output_folder, "graph.json"), "w") as f:
        json.dump(Graph, f)
    with open(os.path.join(args.output_folder, "node_chunk_map.json"), "w") as f:
        json.dump(node_chunk_map, f)

if __name__ == "__main__":
    main()