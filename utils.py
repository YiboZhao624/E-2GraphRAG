from typing import List
from transformers import AutoTokenizer

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