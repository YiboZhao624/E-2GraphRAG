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
import sys

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

def build_tree_pipe(conn, llm_queue, text, cache_path, tokenizer_queue, length, overlap, merge_num):
    llm = llm_queue.get()  # Get llm from queue
    tokenizer = tokenizer_queue.get()
    result = build_tree(text, llm, cache_path, tokenizer, length, overlap, merge_num)
    llm_queue.put(llm)  # Put llm back to queue
    tokenizer_queue.put(tokenizer)
    conn.send(result)
    conn.close()

def extract_graph_pipe(conn, nlp_queue, text, cache_path):
    nlp = nlp_queue.get()  # Get nlp from queue
    result = extract_graph(text, cache_path, nlp)
    nlp_queue.put(nlp)  # Put nlp back to queue
    conn.send(result)
    conn.close()

def parallel_build_extract(text, llm_queue, nlp_queue, tokenizer_queue, cache_path, length, overlap, merge_num):
    timer = Timer()
    build_conn_parent, build_conn_child = mp.Pipe()
    extract_conn_parent, extract_conn_child = mp.Pipe()

    build_tree_timed = timed(timer, "build_tree")(build_tree_pipe)
    extract_graph_timed = timed(timer, "extract_graph")(extract_graph_pipe)
    
    with timer.timer("total"):
        processes = [
            mp.Process(target=build_tree_timed,
                    args=(build_conn_child, llm_queue, text, cache_path, tokenizer_queue, length, overlap, merge_num)),
            mp.Process(target=extract_graph_timed,
                    args=(extract_conn_child, nlp_queue, text, cache_path))
        ]

        for p in processes:
            p.start()

        build_res = build_conn_parent.recv()
        extract_res = extract_conn_parent.recv()

        for p in processes:
            p.join()
    print("-" * 15)
    print(timer.summary())
    print("-" * 15)

    return {
        "build_tree": build_res,
        "extract_graph": extract_res
    }

def main():
    try:
        # parse the arguments.
        configs = parse_args()

        # Create shared queues
        llm_queue = mp.Queue()
        nlp_queue = mp.Queue()
        tokenizer_queue = mp.Queue()
        # Initialize models and put them in queues
        nlp = load_nlp()
        nlp_queue.put(nlp)

        # load the dataset.
        if configs["dataset"]["dataset_name"] == "NovelQA":
            dataset = NovelQALoader(configs["dataset"]["dataset_path"])
        
        elif configs["dataset"]["dataset_name"] == "NarrativeQA":
            dataset = NarrativeQALoader()
        
        elif configs["dataset"]["dataset_name"] == "test":
            dataset = test_loader(configs["dataset"]["dataset_path"])

        else:
            raise ValueError("Invalid dataset")
        # preliminary.
        llm = AutoModel.from_pretrained(configs["llm"]["llm_path"])
        if "Qwen2Model" in str(type(llm)):
            from transformers import Qwen2ForCausalLM
            llm = Qwen2ForCausalLM.from_pretrained(configs["llm"]["llm_path"])

        tokenizer = AutoTokenizer.from_pretrained(configs["llm"]["llm_path"])
        llm_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer, device=configs["llm"]["llm_device"])
        llm_queue.put(llm_pipeline)

        try:
            for data_piece in dataset:
                text = data_piece["book"]
                # first chunk the text book for input of the build_tree.
                text = sequential_split(text, tokenizer, configs["cluster"]["length"], configs["cluster"]["overlap"])
                qa = data_piece["qa"]
                # process the text:
                tree, graph = parallel_build_extract(text, llm_queue, nlp_queue, tokenizer_queue,
                                             configs["paths"]["cache_path"], configs["cluster"]["length"], 
                                             configs["cluster"]["overlap"], configs["cluster"]["merge_num"])
                print("tree and graph done.")
                G, index = graph
                retriever = Retriever(tree, G, index, nlp)
                
                res = []
                # answer the question.
                for qa_piece in qa:
                    question = qa_piece["question"]
                    answer = qa_piece["answer"]
                    
                    model_supplement = retriever.retrieve(question)

                    if configs["dataset"]["dataset_name"] == "NovelQA":
                        input_text = Prompts["QA_prompt_options"].format(question = question,
                                                     model_supplement = model_supplement)
                        # TODO: input the text to the model and get the probs of options.
                        output_logits = llm(input_text).logits[0,-1]
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
                                                     model_supplement = model_supplement)
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

                # save the result.
                res_path = os.path.join(configs["paths"]["answer_path"], f"{data_piece['id']}.json")
                with open(res_path, "w") as f:
                    json.dump(res, f)
        finally:
            # Clean up resources
            while not llm_queue.empty():
                try:
                    model = llm_queue.get_nowait()
                    if hasattr(model, 'to'):
                        model.to('cpu')
                    del model
                except:
                    pass

            while not nlp_queue.empty():
                try:
                    nlp_model = nlp_queue.get_nowait()
                    del nlp_model
                except:
                    pass

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # evaluate the answer.
        total_rl_score = 0
        total_em_score = 0
        for _, answer, output_text in res:
            rl_score = RL_score(answer, output_text)
            em_score = EM_score(answer, output_text)
            print(f"RL_score: {rl_score}, EM_score: {em_score}")
            total_rl_score += rl_score
            total_em_score += em_score
        print(f"Average RL_score: {total_rl_score / len(res)}")
        print(f"Average EM_score: {total_em_score / len(res)}")
        res.append({
            "rl_score": total_rl_score / len(res),
            "em_score": total_em_score / len(res)
        })


        # save the result.
        res_path = os.path.join(configs["paths"]["answer_path"], "result.json")
        with open(res_path, "w") as f:
            json.dump(res, f)

    except Exception as e:
        print(f"Error occurred: {e}")
        # Kill all child processes
        for child in mp.active_children():
            child.terminate()
            child.join()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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