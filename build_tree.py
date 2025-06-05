from typing import List
import json
import os
from transformers import AutoTokenizer, pipeline
from prompt_dict import Prompts
from utils import sequential_split, Timer
import time

def sequential_merge(chunks:List[str], 
                     tokenizer:AutoTokenizer,
                     overlap:int)->str:
    '''
    Merge the chunks into a single text.
    '''
    res = chunks[0]
    for i in range(1, len(chunks)):
        ids = tokenizer(chunks[i], return_tensors="pt")["input_ids"][0][overlap:]
        res += tokenizer.decode(ids)
    return res

def load_cache_summary(cache_path:str)->List[str]:
    '''
    Load the summary from the cache file.
    The cache file is a json file, name as {book_id}_summary.json.
    keys are:
    chunk_id:{
        "text: str,
        "children": List[str],
        "parent": str,
    }
    '''
    with open(cache_path, "r") as f:
        return json.load(f)

def summarize_leaf(text:str, llm:pipeline, language:str)->List[str]:
    '''
    Summarize the text into chunks.
    '''
    if language == "en":
        prompt = Prompts["summarize_details"].format(content=text)
    else:
        prompt = Prompts["summarize_details_zh"].format(content=text)
    res = llm(prompt)[0]["generated_text"][len(prompt):]
    return res

def summarize_summary(text:str, llm:pipeline, language:str)->List[str]:
    '''
    Summarize the summary into chunks.
    '''
    if language == "en":
        prompt = Prompts["summarize_summary"].format(summary=text)
    else:
        prompt = Prompts["summarize_summary_zh"].format(summary=text)
    res = llm(prompt)[0]["generated_text"][len(prompt):]
    return res

def build_tree(text_chunks:List[str], llm:pipeline, cache_folder:str,
               tokenizer:AutoTokenizer, length:int, overlap:int, merge_num:int, language:str):
    '''
    Build the tree from the text.
    '''
    build_start_time = time.time()
    if os.path.exists(os.path.join(cache_folder, "tree.json")):
        return load_cache_summary(os.path.join(cache_folder, "tree.json")), -1

    cache = {}
    
    # leaf ids in the format of "leaf_{i}"
    # due to the leaf has no children, it is set as None.
    for i in range(len(text_chunks)):
        cache["leaf_{}".format(i)] = {
            "text": text_chunks[i],
            "children": None,
            "parent": None,
        }

    # do summarize for the first level.
    summary_id_count = 0
    for i in range(0, len(text_chunks), merge_num):
        merged_chunks = sequential_merge(text_chunks[i:i+merge_num], tokenizer, overlap)
        summary = summarize_leaf(merged_chunks, llm, language)
        cache["summary_0_{}".format(summary_id_count)] = {
            "text": summary,
            "children": [f"leaf_{j}" for j in range(i, i+merge_num)],
            "parent": [],
        }
        summary_id_count += 1
        for j in range(i, min(i+merge_num, len(text_chunks))):
            cache["leaf_{}".format(j)]["parent"]="summary_{}".format(summary_id_count)

    to_summarize = [f"summary_0_{i}" for i in range(summary_id_count)]
    to_summarize = [cache[i]["text"] for i in to_summarize]
    level = 1
    # do summarize for the rest levels.
    while len(to_summarize) > 1.2 * merge_num:
        new_summary_id_count = 0
        for i in range(0, len(to_summarize), merge_num):
            # for summary, there is no overlap.
            merged_chunks = sequential_merge(to_summarize[i:i+merge_num], tokenizer, 0)
            # for summary, using different prompt.
            summary = summarize_summary(merged_chunks, llm, language)
            # key format: summary_{level}_{i}
            cache["summary_{}_{}".format(level, new_summary_id_count)] = {
                "text": summary,
                "children": [f"summary_{level-1}_{j}" for j in range(i, i+merge_num)],
                "parent": [],
            }
            new_summary_id_count += 1
            for j in range(i, min(i+merge_num, len(to_summarize))):
                cache["summary_{}_{}".format(level-1, j)]["parent"] = f"summary_{level}_{new_summary_id_count}"
        # update the to_summarize list.
        to_summarize = [f"summary_{level}_{i}" for i in range(new_summary_id_count)]
        level += 1
    
    # save the cache.
    with open(os.path.join(cache_folder, "tree.json"), "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)
    build_end_time = time.time()
    return_time = build_end_time - build_start_time
    return cache, return_time

if __name__ == "__main__":
    pass