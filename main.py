import multiprocessing as mp
from build_tree import build_tree
from extract_graph import extract_graph
from utils import Timer, timed
from dataloader import NovelQALoader, NarrativeQALoader
import argparse
from transformers import pipeline, AutoTokenizer, AutoModel
from query import Retriever
from prompt_dict import Prompts
import os
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--length", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--merge_num", type=int, default=5)
    parser.add_argument("--answer_path", type=str, default="answer.json")
    parser.add_argument("--llm_device", type=str, default="cuda:4")
    parser.add_argument("--emb_device", type=str, default="cuda:5")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset_path", type=str, default="NovelQA")
    args = parser.parse_args()
    
    return args

def parallel_build_extract(text, llm, tokenizer, cache_path, length, overlap, merge_num):

    timer = Timer()

    build_conn_parent, build_conn_child = mp.Pipe()
    extract_conn_parent, extract_conn_child = mp.Pipe()

    def build_tree_pipe(conn):
        result = build_tree(text, llm, cache_path, tokenizer, length, overlap, merge_num)
        conn.send(result)
        conn.close()

    def extract_graph_pipe(conn):
        result = extract_graph(text, cache_path)
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
    args = parse_args()

    # load the dataset.
    if args.dataset == "NovelQA":
        dataset = NovelQALoader(args.dataset_path)
        llm = AutoModel.from_pretrained(args.llm)
        tokenizer = AutoTokenizer.from_pretrained(args.llm)
        llm_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer, device=args.llm_device)
        
    elif args.dataset == "NarrativeQA":
        dataset = NarrativeQALoader()
        llm = AutoModel.from_pretrained(args.llm)
        tokenizer = AutoTokenizer.from_pretrained(args.llm)
        llm_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer, device=args.llm_device)

    else:
        raise ValueError("Invalid dataset")
    
    # chunk the dataset.

    # preliminary.
    
    


    # for data_piece in dataset:
        # parallel_build_extract(data_piece) - save the cache.
        # answer the question.
        # save the answer.
    for data_piece in dataset:
        text = data_piece["book"]
        qa = data_piece["qa"]
        # process the text:
        tree, graph = parallel_build_extract(text, llm_pipeline, tokenizer,
                                             args.cache_path, args.length, 
                                             args.overlap, args.merge_num)
        retriever = Retriever(tree, graph)
        
        res = []
        # answer the question.
        for qa_piece in qa:
            question = qa_piece["question"]
            answer = qa_piece["answer"]
            
            model_supplement = retriever.retrieve(question)

            if args.dataset == "NovelQA":
                input_text = Prompts["novel_qa_prompt"].format(question = question,
                                                     model_supplement = model_supplement)
                # TODO: input the text to the model and get the probs of options.

            elif args.dataset == "NarrativeQA":
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
        res_path = os.path.join(args.answer_path, f"{data_piece['id']}.json")
        with open(res_path, "w") as f:
            json.dump(res, f)
    # end for loop.


    # end calculate the time.

    # evaluate the answer.

    # save the result.

    # end.

    pass

if __name__ == "__main__":
    main()