import os
import argparse
from GlobalConfig import *
from typing import List
from yb_dataloader import NarrativeQALoader, NovelQALoader
from utils import load_LLM
from tqdm import tqdm
import logging
import json
from transformers import AutoTokenizer, AutoModel, pipeline
from prompt_dict import Prompts


logger = logging.getLogger("build_tree")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def sequential_split(text:str, tokenizer:AutoTokenizer,
                     length:int, overlap:int)->List[str]:
    '''
    Split the text into chunks of length length with overlap.
    '''
    chunks = []
    text_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    for i in range(0, len(text_ids), length - overlap):
        chunk = tokenizer.decode(text_ids[i:i+length])
        chunks.append(chunk)
    return chunks

def sequential_merge(chunks:List[str], 
                     tokenizer:AutoTokenizer,
                     length:int, overlap:int)->str:
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

def summarize_leaf(text:str, llm:pipeline,)->List[str]:
    '''
    Summarize the text into chunks.
    '''
    prompt = Prompts["summarize_details"].format(text=text)
    res = llm(prompt)
    return res

def summarize_summary(text:str, llm:pipeline,)->List[str]:
    '''
    Summarize the summary into chunks.
    '''
    prompt = Prompts["summarize_summary"].format(text=text)
    res = llm(prompt)
    return res

def build_tree(text:str, llm:pipeline, cache_path:str,
               tokenizer:AutoTokenizer, length:int, overlap:int, merge_num:int)->dict:
    '''
    Build the tree from the text.
    '''
    cache = {}
    text_chunks = sequential_split(text, tokenizer, length, overlap)
    
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
        merged_chunks = sequential_merge(text_chunks[i:i+merge_num], tokenizer, length, overlap)
        summary = summarize_leaf(merged_chunks, llm)
        cache["summary_{}".format(summary_id_count)] = {
            "text": summary,
            "children": [f"leaf_{j}" for j in range(i, i+merge_num)],
            "parent": [],
        }
        summary_id_count += 1
        for j in range(i, i+merge_num):
            cache["leaf_{}".format(j)]["parent"]="summary_{}".format(summary_id_count)

    # do summarize for the rest levels.
    

def test():
    tokenizer = AutoTokenizer.from_pretrained(Qwen2_5_14B_Instruct)
    text = "Hello, world! This is a test. This is another test. This is a third test. This is a fourth test. This is a fifth test. This is a sixth test. This is a seventh test. This is a eighth test. This is a ninth test. This is a tenth test."
    length = 10
    overlap = 5
    chunks = sequential_split(text, tokenizer, length, overlap)
    for chunk in chunks:
        assert len(tokenizer(chunk, return_tensors="pt")["input_ids"][0]) <= length,\
            "Test Sequential Split Error. Expected length: {}, Actual length: {}".format(length, len(tokenizer(chunk, return_tensors="pt")["input_ids"][0]))
    print("Test Sequential Split Passed.")
    merged = sequential_merge(chunks, tokenizer, length, overlap)
    assert merged == text, "Test Sequential Merge Error. Expected: {}, Actual: {}".format(text, merged)
    print("Test Sequential Merge Passed.")




if __name__ == "__main__":
    test()

