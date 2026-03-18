import multiprocessing as mp
from extract_graph import load_nlp
from utils import sequential_split, load_dataset, load_tree_graph
import yaml
from transformers import AutoTokenizer
from query import Retriever
from prompt_dict import Prompts
import os
import json
import traceback
import sys
import argparse
import time
from llm_providers import create_llm

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
    # load the dataset.
    dataset = load_dataset(configs["dataset"]["dataset_name"], configs["dataset"].get("dataset_path", None))
    print("dataset loaded!")

    # Load tokenizer for text splitting
    tokenizer = AutoTokenizer.from_pretrained(
        configs["llm"].get("tokenizer_name", configs["llm"]["llm_path"])
    )

    # Load model for QA
    llm = create_llm(configs["llm"])
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

                if configs["dataset"]["dataset_name"] in ("NovelQA", "InfiniteChoice"):
                    input_text = Prompts["QA_prompt_options"].format(
                        question=question, evidence=evidences
                    )
                    option_scores = llm.predict_options(
                        input_text, ["A", "B", "C", "D"]
                    )
                    output_text = max(option_scores, key=option_scores.get)

                elif configs["dataset"]["dataset_name"] == "InfiniteQALoader":
                    input_text = Prompts["QA_prompt_answer"].format(
                        question=question, evidence=model_supplement
                    )
                    output = llm.generate(input_text)
                    output_text = output.text
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
    
    if "llm" in locals():
        llm.cleanup()
        del llm


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
