import multiprocessing as mp
from build_tree import build_tree
from extract_graph import extract_graph, load_nlp
from utils import Timer, timed, sequential_split, RL_score, EM_score
from dataloader import NovelQALoader, NarrativeQALoader, test_loader
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
from process_utils import build_tree_task, extract_graph_task, clean_cuda_memory

def parse_args():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, required=True)
    # parser.add_argument("--llm", type=str, required=True)
    # parser.add_argument("--cache_path", type=str, required=True)
    # parser.add_argument("--tokenizer", type=str, required=True)
    # parser.add_argument("--length", type=int, default=1200)
    # parser.add_argument("--overlap", type=int, default=100)
    # parser.add_argument("--merge_num", type=int, default=5)
    # parser.add_argument("--answer_path", type=str, default="answer.json")
    # parser.add_argument("--llm_device", type=str, default="cuda:4")
    # parser.add_argument("--emb_device", type=str, default="cuda:5")
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--dataset_path", type=str, default="NovelQA")
    # args = parser.parse_args()
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config

def parallel_build_extract(text, configs, cache_folder, length, overlap, merge_num):
    timer = Timer()
    
    with timer.timer("total"):
        with mp.Pool(processes=2) as pool:
            # 准备参数
            build_args = (
                configs["llm"]["llm_path"],
                configs["llm"]["llm_device"],
                text,
                cache_folder,
                configs["llm"]["llm_path"],
                length,
                overlap,
                merge_num
            )
            
            extract_args = (text, cache_folder)
            
            # 异步执行任务
            build_future = pool.apply_async(build_tree_task, (build_args,))
            extract_future = pool.apply_async(extract_graph_task, (extract_args,))
            
            # 获取结果
            try:
                build_res = build_future.get()
            except Exception as e:
                print(f"构建树失败: {e}")
                print(f"错误类型: {type(e).__name__}")
                print(f"详细错误信息: {e.args}")
                import traceback
                print(f"错误堆栈:\n{traceback.format_exc()}")
                raise e
    
            try:
                extract_res = extract_future.get()
            except Exception as e:
                print(f"提取图失败: {e}")
                print(f"错误类型: {type(e).__name__}")
                print(f"详细错误信息: {e.args}")
                import traceback
                print(f"错误堆栈:\n{traceback.format_exc()}")
                raise e
            
    print("-" * 15)
    print(timer.summary())
    print("-" * 15)

    return build_res, extract_res

def main():
    try:
        # parse the arguments.
        configs = parse_args()
        device_id = int(configs["llm"]["llm_device"].split(':')[1]) if ':' in configs["llm"]["llm_device"] else 0

        # load the dataset.
        if configs["dataset"]["dataset_name"] == "NovelQA":
            dataset = NovelQALoader(configs["dataset"]["dataset_path"])

        elif configs["dataset"]["dataset_name"] == "NarrativeQA":
            dataset = NarrativeQALoader()

        elif configs["dataset"]["dataset_name"] == "test":
            dataset = test_loader(configs["dataset"]["dataset_path"])
            
        else:
            raise ValueError("Invalid dataset")

        # Load tokenizer for text splitting
        tokenizer = AutoTokenizer.from_pretrained(configs["llm"]["llm_path"])

        try:
            for i, data_piece in enumerate(dataset):
                text = data_piece["book"]
                text = sequential_split(text, tokenizer, configs["cluster"]["length"], configs["cluster"]["overlap"])
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
                
                # Load model for QA
                llm = AutoModel.from_pretrained(configs["llm"]["llm_path"])
                if "Qwen2Model" in str(type(llm)):
                    from transformers import Qwen2ForCausalLM
                    llm = Qwen2ForCausalLM.from_pretrained(configs["llm"]["llm_path"])
                llm.eval()
                llm_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer, device=configs["llm"]["llm_device"])
                
                try:
                    # Process QA
                    G, index = graph
                    retriever = Retriever(tree, G, index, load_nlp(), **configs["retriever"]["kwargs"])
                    res = []
                    
                    # answer the question.
                    for qa_piece in qa:
                        question = qa_piece["question"]
                        answer = qa_piece["answer"]
                        try:
                            model_supplement = retriever.query(question, **configs["retriever"]["kwargs"])
                            evidences = model_supplement["chunks"]
                            print("len_chunks: ", model_supplement["len_chunks"])
                            print("entities: ", model_supplement["entities"])
                            print("keys: ", model_supplement["keys"])
                            # print(len(evidences))
                            # evidences = "\n".join(evidences)
                            # TODO for debug.
                            
                        except Exception as e:
                            print(f"Error occurred: {e}")
                            print("traceback:")
                            print(traceback.format_exc())
                            raise e

                        if configs["dataset"]["dataset_name"] == "NovelQA" or configs["dataset"]["dataset_name"] == "test":
                            input_text = Prompts["QA_prompt_options"].format(question = question,
                                                        evidence = evidences)
                            # TODO: input the text to the model and get the probs of options.
                            inputs = tokenizer(input_text, return_tensors="pt").to(configs["llm"]["llm_device"])
                            with torch.no_grad():
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

                        elif configs["dataset"]["dataset_name"] == "NarrativeQA":
                            input_text = Prompts["QA_prompt_answer"].format(question = question,
                                                        evidence = model_supplement)
                            output = llm_pipeline(input_text)
                            output_text = output[0]["generated_text"]
                            print(output_text)
                        else:
                            raise ValueError("Invalid dataset")
                        res.append({
                            "question": question,
                            "answer": answer,
                            "output_text": output_text
                        })
                        
                    os.makedirs(configs["paths"]["answer_path"], exist_ok=True)
                    os.makedirs(os.path.join(configs["paths"]["answer_path"],configs["dataset"]["dataset_name"]), exist_ok=True)

                    # Save results
                    res_path = os.path.join(configs["paths"]["answer_path"],configs["dataset"]["dataset_name"], f"book_{i}.json")
                    with open(res_path, "w") as f:
                        json.dump(res, f)
                
                finally:
                    # Clean up QA resources
                    del llm_pipeline
                    del llm
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
        finally:
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error occurred: {e}")
        # Kill all child processes
        for child in mp.active_children():
            child.terminate()
            child.join()
        
        clean_cuda_memory(device_id)
        raise e

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    try:
        main()
    except Exception as e:
        print(f"Program terminated with error: {e}")
        # Ensure all processes are terminated
        for child in mp.active_children():
            child.terminate()
            child.join()
        sys.exit(1)