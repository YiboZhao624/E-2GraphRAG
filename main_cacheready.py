import multiprocessing as mp
from extract_graph import load_nlp
from utils import sequential_split, load_dataset, load_tree_graph
import yaml
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from query import Retriever
from prompt_dict import Prompts
import os
import json
import numpy as np
import traceback
import sys
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    return config

def main():
    print("you are running the code with cache ready!")
    print("please make sure the tree and graph are already built and saved in the cache folder.")
    # parse the arguments.
    configs = parse_args()
    device_id = int(configs["llm"]["llm_device"].split(':')[1]) if ':' in configs["llm"]["llm_device"] else 0

    # load the dataset.
    dataset = load_dataset(configs["dataset"]["dataset_name"], configs["dataset"].get("dataset_path", None))
    print("dataset loaded!")

    # Load tokenizer for text splitting
    tokenizer = AutoTokenizer.from_pretrained(configs["llm"]["llm_path"])

    # Load model for QA
    if configs["dataset"]["dataset_name"] == "NovelQA" or configs["dataset"]["dataset_name"] == "InfiniteChoice":
        if "Qwen2" in configs["llm"]["llm_path"]:
            from transformers import Qwen2ForCausalLM
            llm = Qwen2ForCausalLM.from_pretrained(
                configs["llm"]["llm_path"], 
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
        else:
            llm = AutoModel.from_pretrained(
                configs["llm"]["llm_path"],
                torch_dtype=torch.bfloat16
            )
        llm.eval()
        llm.to(configs["llm"]["llm_device"])
    elif configs["dataset"]["dataset_name"] == "InfiniteQALoader":
        llm = pipeline("text-generation", model=configs["llm"]["llm_path"], tokenizer=tokenizer, device=configs["llm"]["llm_device"])
    else:
        raise ValueError("Invalid dataset")
    print("llm loaded!")
    #########################################################
    ############## start the inference. #####################
    #########################################################
    for i, data_piece in enumerate(dataset):
        if i < configs["resume"]["resumeIndex"]:
            print(f"skipping book {i} because it is less than resumeIndex {configs['resume']['resumeIndex']}")
            continue
        print(f"processing book {i}...")
        text = data_piece["book"]
        text = sequential_split(text, tokenizer, configs["cluster"]["length"], configs["cluster"]["overlap"])
        qa = data_piece["qa"]
        
        piece_name = dataset.available_ids[i]
        cache_folder = os.path.join(configs["paths"]["cache_path"], configs["dataset"]["dataset_name"], str(piece_name))
        if not os.path.exists(cache_folder):
            raise ValueError(f"Cache folder {cache_folder} does not exist.")
        else:
            tree, G, index, appearance_count = load_tree_graph(cache_folder)
        print("tree, G, index, appearance_count loaded!")
        try:
            # Process QA
            if "retriever" not in locals():
                retriever = Retriever(tree, G, index, appearance_count, load_nlp(), **configs["retriever"]["kwargs"])
            else:
                retriever.update(tree, G, index, appearance_count)
            res = []
            os.makedirs(configs["paths"]["answer_path"], exist_ok=True)
            
            # answer the question.
            print(f"start to answer the question...")
            for j, qa_piece in enumerate(qa):
                question = qa_piece["question"]
                answer = qa_piece["answer"]
                
                query_start_time = time.time()
                model_supplement = retriever.query(question, **configs["retriever"]["kwargs"])
                query_end_time = time.time()
                query_time = query_end_time - query_start_time

                with open(os.path.join(configs["paths"]["answer_path"], "query_time.txt"), "a") as f:
                    f.write(f"question {i}: query time: {query_end_time - query_start_time}\n")

                evidences = model_supplement["chunks"]
                print("len_chunks: ", model_supplement.get("len_chunks", 0))
                if model_supplement.get("len_chunks", 0)==0:
                    print(f"TODO:chunk count goes wrong! see book {i} question {j}")
                print("entities: ", model_supplement.get("entities", []))
                print("keys: ", model_supplement.get("keys", []))                
                print("retrieval_type: ", model_supplement.get("retrieval_type", ""))
                count_local = 0
                count_level_n = [0] * 10

                for key, value in model_supplement.get("chunk_ids",{}).items():
                    for chunk_id_supplement in value:
                        if chunk_id_supplement.startswith("leaf"):
                            count_local += 1
                        elif chunk_id_supplement.startswith("summary"):
                            level = int(chunk_id_supplement.split("_")[1])
                            count_level_n[level] += 1
                count_global = sum(count_level_n)
                retrieval_type = model_supplement.get("retrieval_type","Not_recorded.")
                retrieval_chunk_count = model_supplement.get("len_chunks","Not_recorded.")

                if configs["dataset"]["dataset_name"] == "NovelQA" or configs["dataset"]["dataset_name"] == "InfiniteChoice":
                    input_text = Prompts["QA_prompt_options"].format(question = question,evidence = evidences)
                
                    inputs = tokenizer(input_text, return_tensors="pt").to(configs["llm"]["llm_device"])
                    with torch.no_grad():
                        print("inputs token length: ", inputs.input_ids.shape[-1])
                        output_logits = llm(**inputs).logits[0,-1]
            
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

                elif configs["dataset"]["dataset_name"] == "InfiniteQALoader":
                    input_text = Prompts["QA_prompt_answer"].format(question = question,
                                                evidence = model_supplement)
                    output = llm(input_text)
                    output_text = output[0]["generated_text"]
                    output_text = output_text[len(input_text):]
                    print("output_text: ", output_text)
                else:
                    raise ValueError("Invalid dataset")
                res.append({
                    "question": question,
                    "answer": answer,
                    "output_text": output_text,
                    "evidences": qa_piece.get("evidence", None),
                    "type": retrieval_type,
                    "chunk_count": retrieval_chunk_count,
                    "chunk_count_local": count_local,
                    "chunk_count_levels": count_level_n,
                    "chunk_count_global": count_global,
                    "query_time": query_time
                })
                
            os.makedirs(configs["paths"]["answer_path"], exist_ok=True)
            os.makedirs(os.path.join(configs["paths"]["answer_path"],configs["dataset"]["dataset_name"]), exist_ok=True)

            # Save results
            res_path = os.path.join(configs["paths"]["answer_path"],configs["dataset"]["dataset_name"], f"book_{i}.json")
            with open(res_path, "w") as f:
                json.dump(res, f, indent=4)
        
        except Exception as e:
            print(f"Error occurred during QA processing: {e}")
            print("traceback:")
            print(traceback.format_exc())
            print(f"TODO:Error occurred during book {i} processing. Set resumeIndex to {i}.")
            raise e
            
    
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    try:
        main()
    except Exception as e:
        print(f"Program terminated with error: {e}")
        print(traceback.format_exc())
        # Ensure all processes are terminated
        for child in mp.active_children():
            child.terminate()
            child.join(timeout = 3)
        sys.exit(1)
