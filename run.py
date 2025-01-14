'''
1. load data, including the book, summary, mapping, node_chunk_map, triplets, and the question.
2. extract the entities from the question.
3. entities -> chunks by TAG mapping.
4. triplets + summaries + chunks - FILTER-> input knowledge.
5. input knowledge + question -> answer.
6. save the answer into the dataloader.
7. dataloader.calculate_metrics()
'''

import os
import sys
import json
import logging 
import argparse
from typing import List
from yb_dataloader import NovelQALoader, NarrativeQALoader
from prompts import QUERY_PROMPT
from utils import load_LLM
from tqdm import tqdm

logger = logging.getLogger("run")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def query_ner():
    pass

def prepare_question(question:dict) -> str:
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = "NovelQA", choices = ["NovelQA", "NarrativeQA"])
    parser.add_argument("--doc_dir", type=str, default = "./NovelQA")
    parser.add_argument("--model", type=str, default = "")
    args = parser.parse_args()

    model, tokenizer = load_LLM(args.model)
    model.eval()
    model.to(args.device)

    if args.dataset == "NovelQA":
        dataloader = NovelQALoader(saving_folder=args.doc_dir, tokenizer=tokenizer, load_summary_index=True)
    elif args.dataset == "NarrativeQA":
        dataloader = NarrativeQALoader(saving_folder=args.doc_dir, tokenizer=tokenizer, load_summary_index=True)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    for book in tqdm(dataloader, desc = f"Evaluating on Dataset {args.dataset}"):
        questions = book["qa"]

        for question in questions:
            question_text = prepare_question(question)