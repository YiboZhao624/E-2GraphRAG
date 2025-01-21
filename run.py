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
import torch
import logging 
import argparse
import numpy as np
from typing import List
from yb_dataloader import NovelQALoader, NarrativeQALoader
from retriever import TAGRetriever
from prompts import QA_PROMPT
from utils import load_LLM
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger("run")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


def prepare_question(question_id, question,dataset) -> str:
    if dataset == "NovelQA":
        '''
        NovelQA:
        question_id: Q0764
        question: What is the name of the main character?
        Options:
            A: John
            B: Jane
            C: Jim
            D: Jill
        '''
        question_text = question["Question"] + "\n"
        options = question["Options"]
        for option in options:
            question_text += f"{option}: {options[option]}\n"
        return question_text        
    elif dataset == "NarrativeQA":
        '''
        NarrativeQA:
        question_id: What is the name of the main character?
        value: answer of the question.
        '''
        return question_id
    elif dataset == "Lihuaworld":
        raise NotImplementedError("Lihuaworld is not supported yet.")
    else:
        raise ValueError(f"Dataset {dataset} not supported.")



def prepare_chunk_supplement(chunk_supplement:List[dict]) -> str:
    chunk_supplement_text = ""
    for i, chunk in enumerate(chunk_supplement):
        chunk_supplement_text += f"{i+1}. {chunk['text']}\n"
    return chunk_supplement_text

def prepare_graph_supplement(graph_supplement:List[tuple]) -> str:
    '''assert the graph_supplement is a list of tuples, each tuple is a triplet (entity1, entity2, relation_score).'''
    # graph_supplement.sort(key=lambda x: x[2], reverse=True)
    # graph_supplement_text = "The important related relation are:"
    # for triplet in graph_supplement:
    #     graph_supplement_text += f"{triplet[0]}, {triplet[1]}, related score: {triplet[2]}\n"
    graph_supplement_text = "The important related entities are: "
    for entity in graph_supplement:
        graph_supplement_text += f"{entity}, "
    return graph_supplement_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = "NovelQA", choices = ["NovelQA", "NarrativeQA"])
    parser.add_argument("--doc_dir", type=str, default = "./NovelQA")
    parser.add_argument("--model", type=str, default = "")
    parser.add_argument("--embedder_device", type=str, default = "cuda")
    parser.add_argument("--device_ids", type=str, default = "0,1,2,3")
    parser.add_argument("--embed_model", type=str, default = "BAAI/bge-m3")
    parser.add_argument("--embedding_cache_path", type=str, default = "./cache")
    parser.add_argument("--ans_log_folder", type=str, default = "./ans_log")
    args = parser.parse_args()
    
    qa_device = [int(id) for id in args.device_ids.split(",")]
    n_gpu = len(qa_device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map = "auto",
        max_memory = {i: "78GB" for i in range(n_gpu)},
        quantization_config = None,
        trust_remote_code = True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    
    embedding_model = SentenceTransformer(args.embed_model)
    embedding_model.eval()
    embedding_model.to(args.embedder_device)
    if args.dataset == "NovelQA":
        dataloader = NovelQALoader(saving_folder=args.doc_dir, tokenizer=tokenizer, load_summary_index=True)
    elif args.dataset == "NarrativeQA":
        dataloader = NarrativeQALoader(saving_folder=args.doc_dir, tokenizer=tokenizer, load_summary_index=True)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    for book in tqdm(dataloader, desc = f"Evaluating on Dataset {args.dataset}"):
        questions = book["qa"]
        
        Retriever = TAGRetriever(dataloader, book["book_id"], embedding_model, args.embed_model, args.embedder_device, args.embedding_cache_path)
        
        ans_log_folder = os.path.join(args.ans_log_folder, f"{args.dataset}")
        os.makedirs(ans_log_folder, exist_ok=True)
        ans_log = os.path.join(ans_log_folder, f"{book['book_id']}.json")
        
        # answer the questions.
        for question_id, question in questions.items():
            question_text = prepare_question(question_id, question, args.dataset)
            chunk_supplement, graph_supplement, entities = Retriever.query(question_text, book["book_id"])
            chunk_supplement_text = prepare_chunk_supplement(chunk_supplement)
            graph_supplement_text = prepare_graph_supplement(graph_supplement)

            inputs = QA_PROMPT.format(evidence = chunk_supplement_text, important_entities = graph_supplement_text, question = question_text)
            
            if args.dataset == "NarrativeQA":
                inputs = tokenizer(inputs, return_tensors="pt").to(f"cuda:{qa_device[0]}")
                with torch.amp.autocast(device_type = "cuda"):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens = 100,
                        do_sample = True,
                        num_beams = 4,
                        no_repeat_ngram_size = 5,
                        top_p = 0.95,
                        top_k = 60,
                        temperature = 0.7,
                    )
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(inputs):]
            else:
                inputs = tokenizer(inputs, return_tensors="pt").input_ids.to(f"cuda:{qa_device[0]}")
                with torch.amp.autocast(device_type = "cuda"):
                    outputs = model(input_ids = inputs).logits[0, -1]
                
                probs = torch.nn.functional.softmax(
                torch.tensor([
                        outputs[tokenizer("A").input_ids[-1]],
                        outputs[tokenizer("B").input_ids[-1]],
                        outputs[tokenizer("C").input_ids[-1]],
                        outputs[tokenizer("D").input_ids[-1]],
                    ]).float(),
                    dim=0,
                ).detach().cpu().numpy()
                answer = ["A", "B", "C", "D"][np.argmax(probs)]
            # print(answer)
            with open(ans_log, "a") as f:
                f.write(json.dumps({
                    "question_id": question["question_id"],
                    "question": question_text,
                    "answer": answer,
                    "entities in query": entities,
                    "chunk_supplement": [chunk["id"] for chunk in chunk_supplement],
                    "graph_supplement": graph_supplement_text,
                }))
                f.write("\n")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()