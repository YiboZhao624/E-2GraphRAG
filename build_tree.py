import os
import argparse
from GlobalConfig import *
from typing import List
from yb_dataloader import NarrativeQALoader, NovelQALoader
from prompts import SUMMARY_PROMPT
from utils import load_LLM
from tqdm import tqdm
import logging
import json
from transformers import AutoTokenizer
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

