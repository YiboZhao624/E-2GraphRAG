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

def parallel_build_extract(text, llm, tokenizer, cache_path, length, overlap, merge_num, nlp):

    timer = Timer()

    build_conn_parent, build_conn_child = mp.Pipe()
    extract_conn_parent, extract_conn_child = mp.Pipe()

    def build_tree_pipe(conn):
        result = build_tree(text, llm, cache_path, tokenizer, length, overlap, merge_num)
        conn.send(result)
        conn.close()

    def extract_graph_pipe(conn):
        result = extract_graph(text, cache_path, nlp)
        conn.send(result)
        conn.close()

    build_tree_timed = timed(timer, "build_tree")(build_tree_pipe)
    extract_graph_timed = timed(timer, "extract_graph")(extract_graph_pipe)
    
    with timer.timer("total"):
        processes = [
            mp.Process(target=build_tree_timed,
                    args=(build_conn_child,)),
            mp.Process(target=extract_graph_timed,
                    args=(extract_conn_child,))
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
    # parse the arguments.
    configs = parse_args()

    nlp = load_nlp()

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
    tokenizer = AutoTokenizer.from_pretrained(configs["llm"]["llm_path"])
    llm_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer, device=configs["llm"]["llm_device"])
    
    # for data_piece in dataset:
        # parallel_build_extract(data_piece) - save the cache.
        # answer the question.
        # save the answer.
    for data_piece in dataset:
        text = data_piece["book"]
        # first chunk the text book for input of the build_tree.
        text = sequential_split(text, tokenizer, configs["cluster"]["length"], configs["cluster"]["overlap"])
        qa = data_piece["qa"]
        # process the text:
        tree, graph = parallel_build_extract(text, llm_pipeline, tokenizer,
                                             configs["cluster"]["cache_path"], configs["cluster"]["length"], 
                                             configs["cluster"]["overlap"], configs["cluster"]["merge_num"], nlp)
        G, index = graph
        retriever = Retriever(tree, G, index, nlp)
        
        res = []
        # answer the question.
        for qa_piece in qa:
            question = qa_piece["question"]
            answer = qa_piece["answer"]
            
            model_supplement = retriever.retrieve(question)

            if configs["dataset"]["dataset_name"] == "NovelQA":
                input_text = Prompts["novel_qa_prompt"].format(question = question,
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
                input_text = Prompts["narrative_qa_prompt"].format(question = question,
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
    # end for loop.



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

if __name__ == "__main__":
    main()