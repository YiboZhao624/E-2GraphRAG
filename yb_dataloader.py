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
        for root, dirs, files in os.walk(bookpath, topdown=True):
            for directory in dirs:   # Copyright Protected and Public Domain
                for filename in os.listdir(os.path.join(bookpath, directory)):
                    with open(os.path.join(bookpath, directory, filename), "r") as infile:
                        book_id = filename.split('.')[0]
                        dataset[book_id] = {}
                        dataset[book_id]["book"] = infile.read()
        for root, dirs, files in os.walk(datapath, topdown=True):
            for directory in dirs:   # Copyright Protected and Public Domain
                for filename in os.listdir(os.path.join(datapath, directory)):
                    with open(os.path.join(datapath, directory, filename), "r") as infile:
                        qa_id = filename.split('.')[0]
                        dataset[qa_id]["qa"] = json.loads(infile.read())
        return dataset
    
    def __getitem__(self, bid:int):
        '''Get the item by book id.'''
        book_id = self.available_books[bid]
        book_data = self.dataset[book_id]
        result = {
            "book_id": book_id,
            "book": book_data["book"],
            "book_chunks": book_data["book_chunks"],
            "qa": book_data["qa"]
        }
        
        # 添加摘要层如果存在
        if "summary_layers" in book_data:
            result["summary_layers"] = book_data["summary_layers"]
        if "mapping_layers" in book_data:
            result["mapping_layers"] = book_data["mapping_layers"]
        
        return result
    
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

    def update_book_summary(self, book_id, depth, summaries, mappings):
        if "summary_layers" not in self.dataset[self.available_books[book_id]]:
            self.dataset[self.available_books[book_id]]["summary_layers"] = {}
            self.dataset[self.available_books[book_id]]["mapping_layers"] = {}
        
        self.dataset[self.available_books[book_id]]["summary_layers"][depth] = summaries
        self.dataset[self.available_books[book_id]]["mapping_layers"][depth] = mappings

    def load_dataset(self, summary_folder:str, extraction_folder:str):
        '''Load the dataset from the folder.'''
        for file in os.listdir(summary_folder):
            with open(os.path.join(summary_folder, file), "r") as infile:
                book_summary = json.loads(infile.read())
                book_id = file.split('.')[0]
                self.dataset[book_id]["summary_layers"] = book_summary["summary_layers"]
                self.dataset[book_id]["mapping_layers"] = book_summary["mapping_layers"]
        for file in os.listdir(extraction_folder):
            with open(os.path.join(extraction_folder, file), "r") as infile:
                if file.endswith("_node_chunk_map.json"):
                    book_id = file.split('_')[0]
                    node_chunk_mapping = json.loads(infile.read())
                    self.dataset[book_id]["node_chunk_mapping"] = node_chunk_mapping
                else:
                    book_id = file.split('_')[0]
                    extracted_data = json.loads(infile.read())
                    nodes = []
                    relations = []
                    triplets = []
                    for item in extracted_data:
                        nodes.extend(item["nodes"])
                        relations.extend(item["relations"])
                        triplets.extend(item["triplets"])
                    self.dataset[book_id]["nodes"] = nodes
                    self.dataset[book_id]["relations"] = relations
                    self.dataset[book_id]["triplets"] = triplets
        return self.dataset
    
    def save_res(self, ava_book_id, ans):
        book_data = self.dataset[self.available_books[ava_book_id]]
        book_data["answer"] = ans
        self.dataset[self.available_books[ava_book_id]] = book_data
        return self.dataset
    
    def cal_metrics(self):
        '''
        Acc
        '''
        question_count = 0
        correct_count = 0
        for book_id in self.available_books:
            book_data = self.dataset[book_id]
            for i, qa in enumerate(book_data["qa"]):
                question_count += 1
                if qa["Gold"] == book_data["answer"][i]:
                    correct_count += 1
        return correct_count / question_count


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