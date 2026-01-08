import multiprocessing as mp
from extract_graph import load_nlp
from utils import Timer, sequential_split, logger
import yaml
import torch
from transformers import AutoTokenizer
from query import Retriever
from prompt_dict import Prompts
import os
import json
import traceback
import sys
import argparse
import time
from process_utils import build_tree_task, extract_graph_task, clean_cuda_memory
import gc
from datetime import datetime
from utils import load_dataset
from llm_providers import create_llm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    return config

def parallel_build_extract(text, configs, cache_folder, length, overlap, merge_num):
    timer = Timer()
    device_id = int(configs["llm"]["llm_device"].split(':')[1]) if ':' in configs["llm"]["llm_device"] else 0
    
    try:
        with timer.timer("total"):
            with mp.Pool(processes=2) as pool:
                logger.info("Starting parallel processing...")
               
                build_args = (
                    configs["llm"],
                    text,
                    cache_folder,
                    configs["llm"].get("tokenizer_name", configs["llm"]["llm_path"]),
                    length,
                    overlap,
                    merge_num,
                    configs.get("language", "en")
                )
                
                extract_args = (text, cache_folder, configs.get("language", "en"), configs["extractor"]["method"], configs["cluster"]["force_Reextract"])
                
                logger.info("Launching build_tree_task...")
                build_future = pool.apply_async(build_tree_task, (build_args,))
                
                logger.info("Launching extract_graph_task...")
                extract_future = pool.apply_async(extract_graph_task, (extract_args,))
                
                # 获取结果
                try:
                    build_res, build_time_cost = build_future.get()
                except Exception as e:
                    logger.error(f"Tree building failed: {e}")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.error(f"Detailed error information: {e.args}")
                    import traceback
                    logger.error(f"Error stack:\n{traceback.format_exc()}")
                    raise e
        
                try:
                    extract_res, extract_time_cost = extract_future.get()
                except Exception as e:
                    logger.error(f"Graph extraction failed: {e}")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.error(f"Detailed error information: {e.args}")
                    import traceback
                    logger.error(f"Error stack:\n{traceback.format_exc()}")
                    raise e

    except Exception as e:
        logger.error(f"Error occurred in parallel_build_extract: {e}")
        logger.error("traceback:")
        logger.error(traceback.format_exc())
        raise e
    
    finally:
        clean_cuda_memory(device_id)
        gc.collect()

    logger.info("-" * 15)
    logger.info(f"total time: {timer['total']} seconds")
    logger.info(f"build time: {build_time_cost} seconds")
    logger.info(f"extract time: {extract_time_cost} seconds")
    logger.info("-" * 15)
    
    if extract_time_cost != -1 and build_time_cost != -1:
        with open(os.path.join(cache_folder, "time_cost.txt"), "w") as f:
            f.write(f"total time: ||{timer['total']}|| seconds\n")
            f.write(f"build time: ||{build_time_cost}|| seconds\n")
            f.write(f"extract time: ||{extract_time_cost}|| seconds\n")
        
    return build_res, extract_res

def main():
    try:
        date = datetime.now().strftime("%Y%m%d")
        # parse the arguments.
        configs = parse_args()
        device_id = int(configs["llm"]["llm_device"].split(':')[1]) if ':' in configs["llm"]["llm_device"] else 0

        # load the dataset.
        dataset = load_dataset(configs["dataset"]["dataset_name"], configs["dataset"]["dataset_path"])

        # Load tokenizer for text splitting
        tokenizer = AutoTokenizer.from_pretrained(
            configs["llm"].get("tokenizer_name", configs["llm"]["llm_path"])
        )

        try:
            for i, data_piece in enumerate(dataset):
                if i < configs["resume"]["resumeIndex"]:
                    continue

                text = data_piece["book"]
                if configs.get("split_method", "sequential") == "sequential":
                    text = sequential_split(text, tokenizer, configs["cluster"]["length"], configs["cluster"]["overlap"])
                elif configs.get("split_method", "sequential") == "nn":
                    logger.info("split_method: nn")
                    text = text.split("\n\n")
                elif configs.get("split_method", "sequential") == "manual":
                    logger.info("split_method: manual")
                    text = text.split("####")
                qa = data_piece["qa"]
                
                piece_name = dataset.available_ids[i]
                cache_folder = os.path.join(configs["paths"]["cache_path"], configs["dataset"]["dataset_name"], str(piece_name))
                if not os.path.exists(cache_folder):
                    os.makedirs(cache_folder)

                # Process with parallel execution
                tree, graph = parallel_build_extract(
                    text, configs, cache_folder,
                    configs["cluster"]["length"], configs["cluster"]["overlap"],
                    configs["cluster"]["merge_num"]
                )
                
                # Load model for QA using unified backend
                llm = create_llm(configs["llm"])

                try:
                    # Process QA
                    G, index, appearance_count = graph
                    if "retriever" not in locals():
                        retriever = Retriever(tree, G, index, appearance_count, load_nlp(configs["extractor"]["language"], configs["extractor"]["method"]), **configs["retriever"]["kwargs"])
                    else:
                        retriever.update(tree, G, index, appearance_count)
                    res = []
                    os.makedirs(configs["paths"]["answer_path"], exist_ok=True)
                    
                    # answer the question.
                    for j, qa_piece in enumerate(qa):
                        question = qa_piece["question"]
                        answer = qa_piece["answer"]
                        try:
                            query_start_time = time.time()
                            model_supplement = retriever.query(question, **configs["retriever"]["kwargs"])
                            query_end_time = time.time()
                            with open(os.path.join(configs["paths"]["answer_path"], f"{date}_query_time.txt"), "a") as f:
                                f.write(f"question {i}: query time: {query_end_time - query_start_time}\n")

                            evidences = model_supplement["chunks"]
                            logger.info(f"len_chunks: {model_supplement.get('len_chunks', 0)}")
                            logger.info(f"entities: {model_supplement.get('entities', [])}")
                            logger.info(f"keys: {model_supplement.get('keys', [])}")
                            
                        except Exception as e:
                            logger.error(f"Error occurred: {e}")
                            logger.error("traceback:")
                            logger.error(traceback.format_exc())
                            raise e

                        if configs["dataset"]["dataset_name"] in ("NovelQA", "InfiniteChoice"):
                            input_text = Prompts["QA_prompt_options"].format(
                                question=question, evidence=evidences
                            )
                            try:
                                option_scores = llm.predict_options(
                                    input_text, ["A", "B", "C", "D"]
                                )
                                output_text = max(option_scores, key=option_scores.get)
                            except NotImplementedError as e:
                                logger.error(
                                    "Selected LLM backend does not support option scoring."
                                )
                                raise e
                        else: # default as QA.
                            if configs.get("language", "en") == "zh":
                                input_text = Prompts["QA_prompt_answer_zh"].format(
                                    question=question, evidence=model_supplement
                                )
                            else:
                                input_text = Prompts["QA_prompt_answer"].format(
                                    question=question, evidence=model_supplement
                                )
                            logger.info(f"input_text: {len(input_text)}")
                            output = llm.generate(input_text, max_new_tokens=300)
                            output_text = output.text
                            logger.info(f"output_text: {output_text}")
                        res.append({
                            "question": question,
                            "answer": answer,
                            "output_text": output_text,
                            "evidences": qa_piece.get("evidence", None)
                        })
                        
                    os.makedirs(configs["paths"]["answer_path"], exist_ok=True)
                    os.makedirs(os.path.join(configs["paths"]["answer_path"],configs["dataset"]["dataset_name"]), exist_ok=True)

                    # Save results
                    res_path = os.path.join(configs["paths"]["answer_path"],configs["dataset"]["dataset_name"], f"book_{i}.json")
                    with open(res_path, "w") as f:
                        json.dump(res, f, indent=4)
                
                except Exception as e:
                    logger.error(f"Error occurred during QA processing: {e}")
                    logger.error("traceback:")
                    logger.error(traceback.format_exc())
                    logger.error(f"TODO:Error occurred during book {i} processing. Set resumeIndex to {i}.")
                    raise e
                finally:
                    if "llm" in locals():
                        llm.cleanup()
                        del llm
                        logger.info("llm deleted")
                
        except Exception as e:
            logger.error(f"Error occurred during dataset processing: {e}")
            logger.error("traceback:")
            logger.error(traceback.format_exc())
            raise e
        finally:
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error occurred in main: {e}")
        logger.error("traceback:")
        logger.error(traceback.format_exc())
        # Ensure all processes are terminated
        for child in mp.active_children():
            child.terminate()
            child.join(timeout=3)
        clean_cuda_memory(device_id)
        raise e

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    try:
        main()
    except Exception as e:
        logger.error(f"Program terminated with error: {e}")
        logger.error(traceback.format_exc())
        # Ensure all processes are terminated
        for child in mp.active_children():
            child.terminate()
            child.join(timeout = 3)
        sys.exit(1)