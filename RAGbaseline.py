from dataloader import NovelQALoader, NarrativeQALoader
from utils import sequential_split, EM_score, RL_score
import yaml, argparse, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, pipeline
from prompt_dict import Prompts
import numpy as np
import os,json,pickle
import faiss
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)
    return configs

def build_faiss(text, embedder:SentenceTransformer, cache_path) -> faiss.IndexFlatIP:
    embeddings = embedder.encode(text)
    with open(os.path.join(cache_path, "embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
    return faiss_index


def main():
    configs = parse_args()
    # load the dataset.
    if configs["dataset"]["dataset_name"] == "NovelQA":
        dataset = NovelQALoader(configs["dataset"]["dataset_path"])
    elif configs["dataset"]["dataset_name"] == "NarrativeQA":
        dataset = NarrativeQALoader()
    else:
        raise ValueError("Invalid dataset")
    
    # load the model.
    if configs["dataset"]["dataset_name"] == "NovelQA":
        llm = AutoModel.from_pretrained(configs["model"]["model_path"])
        tokenizer = AutoTokenizer.from_pretrained(configs["model"]["tokenizer_path"])
        llm.eval()
        llm.to(configs["model"]["device"])
    elif configs["dataset"]["dataset_name"] == "NarrativeQA":
        tokenizer = AutoTokenizer.from_pretrained(configs["model"]["tokenizer_path"])
        llm = pipeline("text-generation", model=configs["model"]["model_path"], tokenizer=tokenizer, device=configs["model"]["device"])
    else:
        raise ValueError("Invalid dataset")
    

    # load the embedder.
    embedder = SentenceTransformer(configs["embedder"]["embedder_path"])
    embedder.eval()
    embedder.to(configs["embedder"]["device"])

    # sequential split the text.
    total_res = []
    total_build_time = 0
    build_num = 0
    total_search_time = 0
    search_num = 0
    for i, data_piece in enumerate(dataset):
        print(f"Processing book {i}...")
        questions = data_piece["qa"]
        book = data_piece["book"]
        # sequential split the context.
        context_splits = sequential_split(book, tokenizer,1200, 100)
        start_time = time.time()
        faiss_index = build_faiss(context_splits, embedder, configs["paths"]["cache_path"])
        end_time = time.time()
        print(f"Time taken to build the faiss index: {end_time - start_time} seconds")
        total_build_time += end_time - start_time
        build_num += 1
        res = []
        for qa in questions:
            question = qa["question"]
            answer = qa["answer"]
            search_content = question.split("\n")[0]
            # search the faiss index.
            start_time = time.time()
            search_embedding = embedder.encode(search_content)
            # Reshape search_embedding to 2D array
            search_embedding = search_embedding.reshape(1, -1)
            _, indices = faiss_index.search(search_embedding, configs["search"]["top_k"])
            search_chunks = [context_splits[i] for i in indices[0]]  # Use indices from results[1]
            end_time = time.time()
            print(f"Time taken to search the faiss index: {end_time - start_time} seconds")
            total_search_time += end_time - start_time
            search_num += 1
            # get the context.
            context = "\n".join(search_chunks)
            
            if configs["dataset"]["dataset_name"] == "NovelQA":
                llm_input = Prompts["QA_prompt_options"].format(question=question, evidence=context)
                llm_input = tokenizer(llm_input, return_tensors="pt").to(configs["model"]["device"])
                with torch.no_grad():
                    output = llm(**llm_input)
                    output_logits = output.last_hidden_state[0,-1]
                probs = torch.nn.functional.softmax(
                torch.tensor([
                        output_logits[tokenizer("A").input_ids[-1]],
                        output_logits[tokenizer("B").input_ids[-1]],
                        output_logits[tokenizer("C").input_ids[-1]],
                        output_logits[tokenizer("D").input_ids[-1]],
                    ]).float(),
                    dim=0,
                ).detach().cpu().numpy()
                output_text = ["A", "B", "C", "D"][np.argmax(probs)]
            elif configs["dataset"]["dataset_name"] == "NarrativeQA":
                llm_input = Prompts["QA_prompt_answer"].format(question=question, evidence=context)
                output = llm(llm_input)
                output_text = output[0]["generated_text"]
                print(output_text)
            else:
                raise ValueError("Invalid dataset")
            res.append({
                "question": question,
                "answer": answer,
                "output_text": output_text
            })
        res_folder = os.path.join(configs["paths"]["answer_path"],configs["dataset"]["dataset_name"])
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_path = os.path.join(res_folder, f"book_{i}.json")
        with open(res_path, "w") as f:
            json.dump(res, f, indent=4)
        total_res.append(res)
    # calculate the EM and RL score.
    total_em_score = 0
    total_rl_score = 0
    total_num = 0

    for i, data_piece in enumerate(total_res):
        for qa in data_piece:
            answer = qa["answer"]
            output_text = qa["output_text"]
            total_em_score += EM_score(output_text, answer)
            total_rl_score += RL_score(output_text, answer)
            total_num += 1
    print(f"EM score: {total_em_score/total_num}, RL score: {total_rl_score/total_num}")
    print(f"Average time taken to build the faiss index: {total_build_time/build_num} seconds")
    print(f"Average time taken to search the faiss index: {total_search_time/search_num} seconds")


if __name__ == "__main__":
    main()