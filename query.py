'''
1. load data, including the book, summary, mapping, node_chunk_map, triplets, and the question.
2. extract the entities from the question.
3. entities -> chunks by TAG mapping.
4. chunks -> triplets 
5. chunks -> summaries. By Tree Summarization.
6. triplets + summaries + chunks -FILTER-> input knowledge.
7. input knowledge + question -> answer.
8. save the answer into the dataloader.
9. dataloader.calculate_metrics()
'''

import os
import sys
import json
import logging 
import argparse
from typing import List
from yb_dataloader import NovelQALoader
from prompts import NER_PROMPT
from utils import load_LLM

def prepare_data(tokenizer, args)->NovelQALoader:
    # load data
    doc_dir = os.path.join(args.data_dir, "Books")
    qa_dir = os.path.join(args.data_dir, "Data")
    dataloader = NovelQALoader(doc_dir, qa_dir, tokenizer, args.chunk_size, args.overlap)
    dataloader.load_dataset(args.summary_dir, args.extraction_dir)
    return dataloader

def extract_entities(question, model, tokenizer)->List[str]:
    prompt = NER_PROMPT.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    entities = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    entities = entities.split(",")
    entities = [entity.strip("[").strip("]").strip() for entity in entities]
    return entities

def entities_to_chunks(entities, book_nodes, book_node_chunk_map:dict)->List[int]:
    chunks = []
    for entity in entities:
        if book_node_chunk_map.get(entity):
            chunks.append(set(book_node_chunk_map[entity]))
        else:
            chunks.append(set())

    if not chunks:
        return []
    # only when all the entities are mentioned in the chunk, the chunk is valid.
    common_chunks = set.intersection(*chunks)

    return list(common_chunks)

def chunks_to_triplets(chunks, book_triplets):
    triplets = []
    for chunk in chunks:
        triplets.extend(book_triplets[chunk])
    return triplets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./NovelQA/", help="The directory to store the data.")
    parser.add_argument("--output_dir", type=str, default="output", help="The directory to store the output.")
    parser.add_argument("--chunk_size", type=int, default=1200, help="The chunk size for the chunks.")
    parser.add_argument("--overlap", type=int, default=100, help="The overlap for the chunks.")
    parser.add_argument("--summary_dir", type=str, default="summary", help="The directory to store the summaries.")
    parser.add_argument("--extraction_dir", type=str, default="extraction", help="The directory to store the extractions.")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="The model to use.")
    parser.add_argument("--max_length", type=int, default=512, help="The maximum length of the input.")
    parser.add_argument("--num_beams", type=int, default=5, help="The number of beams to use.")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of sequences to return.")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of sequences to return.")
    args = parser.parse_args()
    model, tokenizer = load_LLM(args.model)
    model.to(args.device).eval()
    
    dataloader = prepare_data(tokenizer, args)

    for book in dataloader:
        book_id = book["book_id"]
        book_chunks = book["chunks"]
        book_nodes = book["nodes"]
        book_relations = book["relations"]
        book_triplets = book["triplets"]
        book_summary = book["summary"]
        book_mapping = book["mapping"]
        book_node_chunk_map = book["node_chunk_map"]
        book_summary_layers = book["summary_layers"]
        book_mapping_layers = book["mapping_layers"]
        book_question = book["qa"]
        for QID, question_dict in book_question.items():
            question = question_dict["Question"]
            # preload the ground truth for further analysis.
            ground_truth_context = question_dict["Answer"]
            ground_truth_option = question_dict["Gold"]
            ground_truth_evidences = question_dict["Evidences"]
            # the options for the question. Also can be QA without options.
            # TODO: make it to be a QA without options.
            options = question_dict["Options"]
            options_input = "\n".join([f"{option}: {options[option]}" for option in options])

            # extract the entities from the question.
            entities : List[str] = extract_entities(question, model, tokenizer)

            # entities -> candidate chunks by TAG mapping.
            candidate_chunks = entities_to_chunks(entities, book_nodes, book_node_chunk_map)
            # TODO: get the summary chunks by Tree structure.
            summary_chunks = []



    


if __name__ == "__main__":
    main()