'''
1. format different dataset into a unified format.
2. chunk the text into chunks.
'''

import argparse
import json
import os
import re

from tqdm import tqdm
from transformers import AutoTokenizer

class YBDataLoader:
    '''a general dataloader for lightTAG.'''
    def __init__(self, 
                 docpath:str = "CollectedBooks", 
                 qapath:str = "CollectedData", 
                 ) -> None:
        ''' Initialize the dataloader.
        Args:
            docpath: path to the documents.
            qapath: path to the QA data.
        '''
        self.docpath = docpath
        self.qapath = qapath
    
class NovelQALoader(YBDataLoader):
    '''a dataloader for NovelQA.'''
    def __init__(self, 
                 docpath:str = "CollectedBooks", 
                 qapath:str = "CollectedData", 
                 tokenizer:AutoTokenizer = None,
                 chunk_size:int = 512,
                 overlap:int = 128,
                 ) -> None:
        super().__init__(docpath, qapath)
        self.dataset = self.build_dataset(datapath=qapath, bookpath=docpath)
        self.available_books = list(self.dataset.keys())
        self.available_books.sort()
        self.tokenizer = tokenizer
        self.dataset = self._chunk_book(self.tokenizer, chunk_size=chunk_size, overlap=overlap)

    def build_dataset(self, datapath:str, bookpath:str):
        '''Build the dataset.'''
        dataset = {}
        print("Loading books...")
        for filename in os.listdir(bookpath):
            book_id = filename.split('.')[0]
            # print(f"Found book: {book_id}")
            with open(os.path.join(bookpath, filename), "r") as infile:
                    dataset[book_id] = {}
                    dataset[book_id]["book"] = infile.read()

        print("Loading QA data...")
        for root, dirs, files in os.walk(datapath, topdown=True):
            for filename in os.listdir(datapath):
                qa_id = filename.split('.')[0]
                # print(f"Found QA: {qa_id}")
                if qa_id not in dataset:
                    # print(f"Warning: QA file {qa_id} has no corresponding book!")
                    continue
                with open(os.path.join(datapath, filename), "r") as infile:
                    dataset[qa_id]["qa"] = json.loads(infile.read())
        return dataset
    
    def __getitem__(self, bid:str):
        '''Get the item by book id.'''
        return self.dataset[self.available_books[bid]]
    
    def __len__(self):
        '''Get the length of the dataset.'''
        return len(self.available_books)

    def _chunk_book(self, tokenizer, chunk_size:int = 512, overlap:int = 128):
        '''Chunk the book. Save as token ids.'''
        for bid in tqdm(self.available_books):
            book = self.dataset[bid]["book"]
            book_chunks = []
            tokens = tokenizer(book, return_tensors="pt")
            stride = chunk_size - overlap
            for i in range(0, len(tokens["input_ids"][0]), stride):
                end_idx = min(i + chunk_size, len(tokens["input_ids"][0]))
                chunk_ids = tokens["input_ids"][0][i:end_idx].tolist()
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                book_chunks.append(chunk_text)
            self.dataset[bid]["book_chunks"] = book_chunks

        return self.dataset


class NarrativeQALoader(YBDataLoader):
    '''a dataloader for NarrativeQA.'''
    def __init__(self, 
                 docpath:str = "NarrativeQA/Books", 
                 qapath:str = "NarrativeQA/Data", 
                 ) -> None:
        super().__init__(docpath, qapath)
        raise NotImplementedError("NarrativeQA is not implemented yet.")






if __name__ == "__main__":
    dataloader = NovelQALoader(docpath="NovelQA/Books", qapath="NovelQA/Data/PublicDomain")
    print(len(dataloader))
    print(dataloader[0]["book_chunks"][0])