'''
1. format different dataset into a unified format.
2. chunk the text into chunks.
'''

import argparse
import json
import os
import re

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
                 tokenizer_name:str = "gpt2",
                 chunk_size:int = 512,
                 overlap:int = 128,
                 ) -> None:
        super().__init__(docpath, qapath)
        self.dataset = self.build_dataset(datapath=qapath, bookpath=docpath)
        self.available_books = list(self.dataset.keys())
        self.available_books.sort()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = self.chunk_book(self.tokenizer, chunk_size=chunk_size, overlap=overlap)

    def build_dataset(self, datapath:str, bookpath:str):
        '''Build the dataset.'''
        dataset = {}
        for root, dirs, files in os.walk(bookpath, topdown=True):
            for directory in dirs:
                for filename in os.listdir(os.path.join(bookpath, directory)):
                    with open(os.path.join(bookpath, directory, filename), "r") as infile:
                        dataset[filename[:-4]] = {}
                        dataset[filename[:-4]]["book"] = infile.read()
        for root, dirs, files in os.walk(datapath, topdown=True):
            for filename in os.listdir(datapath):
                with open(os.path.join(datapath, filename), "r") as infile:
                    dataset[filename[:-5]]["qa"] = json.loads(infile.read())
        return dataset
    
    def __getitem__(self, bid:str):
        '''Get the item by book id.'''
        return self.dataset[self.available_books[bid]]
    
    def __len__(self):
        '''Get the length of the dataset.'''
        return len(self.available_books)

    def chunk_book(self, tokenizer, chunk_size:int = 512, overlap:int = 128):
        '''Chunk the book. Save as token ids.'''
        for bid in self.available_books:
            book = self.dataset[bid]["book"]
            book_chunks = []
            tokens = tokenizer(book, return_tensors="pt")
            stride = chunk_size - overlap
            for i in range(0, len(tokens["input_ids"][0]), stride):
                end_idx = min(i + chunk_size, len(tokens["input_ids"][0]))
                book_chunks.append(tokens["input_ids"][0][i:end_idx].tolist())
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