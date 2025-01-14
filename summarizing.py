'''
1. load data
2. clustering method
   2.1. sequential + hyperparameter
   2.2. sequential + model split
   2.3. model clustering by representation
   2.4. model clustering by extracted triplets
3. summarize by LLM.
Recursively execute the 2 and 3.
'''
import os
import argparse
from typing import List
from yb_dataloader import NarrativeQALoader, NovelQALoader
from prompts import SUMMARY_PROMPT
from utils import load_LLM
from tqdm import tqdm
import logging
import json

logger = logging.getLogger("summarize")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def sequential(loader, book_id, tokenizer, args, current_depth)->List[str]:
    '''
    Required args:
        -- merge_num: the number of chunks to merge. if not provided, default to 5.

    '''

    logger.info(f"Sequential summarizing book {book_id} at depth {current_depth}.")
    logger.info(f"book: {book_id}, bid: {loader[book_id]['book_id']}, keys: {loader[book_id].keys()}")
    if args.merge_num is None:
        args.merge_num = 5
        logger.warning(f"Merge number not specified, using default value 5.")
    else:
        merge_num = args.merge_num

    if current_depth == 0:
        book = loader[book_id]
        # inside a book. once a book.
        book_chunks : List[str] = book["book_chunks"]
        merged_book_chunks : List[str] = []
        book_mapping : List[List[int]] = []

        for i in range(0, len(book_chunks), merge_num):
            # inside a cluster.
            chunk = book_chunks[i:i+merge_num]
            merged_chunk = []
            book_chunk_mapping = []
        
            # deduplicate.
            for j, chunk_data in enumerate(chunk):
                global_chunk_idx = i + j
                c = chunk_data["text"]
                if j == 0:
                    merged_chunk.append(c)
                else:
                    # delete the overlap part.
                    c = tokenizer.decode(tokenizer(c, return_tensors="pt")["input_ids"][0][args.overlap:], skip_special_tokens=True)
                    merged_chunk.append(c)
                book_chunk_mapping.append(global_chunk_idx)
            merged_text = "".join(merged_chunk)
            # add the cluster to the book.
            merged_book_chunks.append(merged_text)
            book_mapping.append(book_chunk_mapping)
        return merged_book_chunks, book_mapping

    else:
        logger.info(f"Sequential summarizing book {book_id} at depth {current_depth}.")
        logger.info(f"book: {book_id}, bid: {loader[book_id]['book_id']}, keys: {loader[book_id].keys()}")
        book = loader[book_id]
        if book.get("summary_layers", None) is None:
            raise ValueError(f"Summary layers not found for book {book_id}.")

        # get the book chunks data, it is a List[Dict]
        book_chunks = book["summary_layers"][current_depth-1]
        # get the text of each chunk. List[str]
        book_chunks = [chunk["text"] for chunk in book_chunks]

        book_mapping = []
        merged_book_chunks = []

        for i in range(0, len(book_chunks), merge_num):
            chunk_text = "".join(book_chunks[i:i+merge_num])
            indices = list(range(i, min(i+merge_num, len(book_chunks))))
            book_mapping.append(indices)
            merged_book_chunks.append(chunk_text)
        return merged_book_chunks, book_mapping

def representation(loader, rep_model, args)->List[List[str]]:
    pass

def model_split(loader, model, tokenizer, args)->List[List[str]]:
    pass

def triplets(loader, model, tokenizer, args)->List[List[str]]:
    pass

def recursive_summarize(loader: NovelQALoader, book_id, model, tokenizer, args, max_depth=3, current_depth=0):
    """Recursive summarizing the text.
    Args:
        loader: the data loader or the text list.
        book_id: the book id.
        model: the language model for summarizing.
        tokenizer: the tokenizer.
        args: the arguments.
        max_depth: the max depth of the recursion.
        current_depth: the current depth of the recursion.
    Returns:
        the final summary result and the mapping of each layer for one book.
        the loader will also be changed.
    """
    logger.info(f"Start summarizing the {current_depth + 1} layer.")
    
    # 1. base case: if only one chunk or reach the max depth, return directly.
    if current_depth >= max_depth:
        return 

    # 1. clustering 
    match args.clustering_method:
        case "sequential":
            merged_chunks, mapping = sequential(loader, book_id, tokenizer, args, current_depth)
        case "model_split":
            merged_chunks, mapping = model_split(loader, book_id, model, tokenizer, args, current_depth)
        case "representation":
            merged_chunks, mapping = representation(loader, args, current_depth)
        case "triplets":
            merged_chunks, mapping = triplets(loader, model, tokenizer, args, current_depth)
        case _:
            raise ValueError(f"unsupported clustering method: {args.clustering_method}")

    ############################################################
    # merged_chunks: [
    # [chunk1:3,chunk4:6......],
    # mapping: [
    # [[1,2,3],[4,5,6],.....], -> chunk1:3,chunk4:6.....
    # mapping is a pointer to the original chunks in the former layer.
    ############################################################

    # 2. summarize each chunk.
    book_summaries = []

    for chunk in tqdm(merged_chunks, desc=f"Depth {current_depth} Summarizing"):
        prompt = SUMMARY_PROMPT.format(text=chunk)
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        outputs = model.generate(**inputs, max_new_tokens=1024)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        book_summaries.append(summary)

    # save the current layer result to loader using loader's method
    loader.update_book_summary(book_id, current_depth, book_summaries, mapping)

    # recursively process the next layer.
    if len(book_summaries) > 1:
        final_summary, next_mappings = recursive_summarize(loader, book_id, model, tokenizer, args, max_depth=max_depth, current_depth=current_depth+1)
        return final_summary, [mapping] + next_mappings
    else:
        return book_summaries[0], [mapping]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="data/NovelQA/NovelQA_1200chunksize_summarized.json")
    parser.add_argument("--clustering_method", type=str, default="sequential", choices=["sequential", "model_split", "representation", "triplets"])
    parser.add_argument("--model_name", type=str, default="Qwen2.5-14B-Instruct")
    parser.add_argument("--dataset", type=str, default="NovelQA", choices=["NovelQA", "NarrativeQA"])
    parser.add_argument("--rep_model", type=str, default="Sentence-Bert")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--max_chunks_per_group", type=int, default=10)
    parser.add_argument("--max_summary_depth", type=int, default=5)
    parser.add_argument("--merge_num", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=1200)
    args = parser.parse_args()

    #load LLM
    model, tokenizer = load_LLM(args.model_name)
    model.to(args.device)
    model.eval()

    # load data
    if args.dataset == "NovelQA":
        loader = NovelQALoader(docpath="./NovelQA/Books", qapath="./NovelQA/Data", tokenizer=tokenizer, chunk_size=1200, overlap=args.overlap)
    elif args.dataset == "NarrativeQA":
        loader = NarrativeQALoader(tokenizer=tokenizer, chunk_size=1200, overlap=args.overlap, saving_folder="./NarrativeQA")
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # execute the recursive summarizing.
    for i in tqdm(range(len(loader)), total=len(loader), desc="Summarizing"):
        final_summaries, all_mappings = recursive_summarize(loader, i, model, tokenizer, args, max_depth=args.max_summary_depth)

        # all mapping is useless.
        # save the result to json
        with open(os.path.join(args.output_path, f"{args.dataset}_{loader[i]['book_id']}_chunksize_summarized.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'summary_layers': loader[i]["summary_layers"],
                'mapping_layers': loader[i]["mapping_layers"],
                'args': vars(args)
            }, f, ensure_ascii=False, indent=2)

    # the result has been saved in the loader's properties:
    # loader.summary_layers - the summaries of each layer.
    # loader.mapping_layers - the mapping of each layer.


if __name__ == "__main__":
    main()