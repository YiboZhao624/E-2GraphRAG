'''
1. format different dataset into a unified format.
2. chunk the text into chunks.
'''

import argparse
import json
import os
import re

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Set, Tuple, TypedDict

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("DataLoader")


class chunk_index(TypedDict):
    global_nouns: Set[str]
    chunk_to_nouns: Dict[int, Set[str]]
    noun_to_chunks: Dict[str, Set[int]]
    noun_pairs: Dict[Tuple[str, str], int]

class AbstractDataLoader:
    '''A general dataloader for text data with chunking and hierarchical summary functionality'''
    def __init__(self) -> None:
        self.dataset = {}
        self.available_book_ids = []
        self.tree_structure = {}
        self._index = chunk_index() 
        
    def __getitem__(self, bid:int):
        '''Get the item by book id'''
        book_id = self.available_book_ids[bid]
        book_data = self.dataset[book_id]
        result = {
            "book_id": book_id,
            "book": book_data["book"],
            "book_chunks": book_data["book_chunks"],
            "qa": book_data["qa"],
            "index": self._index[book_id]   
        }
        
        if "summary_layers" in book_data:
            result["summary_layers"] = book_data["summary_layers"]
        if "mapping_layers" in book_data:
            result["mapping_layers"] = book_data["mapping_layers"]
            
        return result
    
    def __len__(self):
        return len(self.available_book_ids)

    def _chunk_book(self, tokenizer:AutoTokenizer, chunk_size:int = 1200, overlap:int = 100):
        '''Chunk books into smaller pieces with overlap'''
        for bid in tqdm(self.available_book_ids, desc = "Chunking Books"):
            book = self.dataset[bid]["book"]
            book_chunks = []
            self.tree_structure[bid] = {
                "nodes": {},
                "children": {},
                "parents": {}
            }
            tokens = tokenizer(book, return_tensors="pt")
            stride = chunk_size - overlap
            chunk_idx = 0
            
            for i in range(0, len(tokens["input_ids"][0]), stride):
                end_idx = min(i + chunk_size, len(tokens["input_ids"][0]))
                token_ids = tokens["input_ids"][0][i:end_idx].tolist()
                chunk_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                chunk_id = f"{bid}_leaf_{chunk_idx}"
                chunk_idx += 1
                
                chunk_data = {
                    "id": chunk_id,
                    "text": chunk_text,
                    "type": "leaf"
                }
                book_chunks.append(chunk_data)
                
                self.tree_structure[bid]["nodes"][chunk_id] = {
                    "text": chunk_text,
                    "level": "leaf",
                    "type":"chunk"
                }
                self.tree_structure[bid]["children"][chunk_id] = []
                self.tree_structure[bid]["parents"][chunk_id] = []
                
            self.dataset[bid]["book_chunks"] = book_chunks

        return self.dataset

    def update_book_summary(self, book_id, depth, summaries, mappings):
        '''Update book summary and mapping information'''
        book_key = self.available_book_ids[book_id]
        if "summary_layers" not in self.dataset[book_key]:
            self.dataset[book_key]["summary_layers"] = {}
            self.dataset[book_key]["mapping_layers"] = {}
        
        summary_chunks = []
        for i, summary in enumerate(summaries):
            summary_id = f"{book_key}_summary_{depth}_{i}"
            summary_data = {
                "id": summary_id,
                "text": summary,
                "type": "summary",
            }
            summary_chunks.append(summary_data)

            self.tree_structure[book_key]["nodes"][summary_id] = {
                "text": summary,
                "level": f"depth_{depth}",
                "type": "summary",
            }
            self.tree_structure[book_key]["children"][summary_id] = []
            self.tree_structure[book_key]["parents"][summary_id] = []

        update_mappings = []
        for i, mapping in enumerate(mappings):
            parent_id = f"{book_key}_summary_{depth}_{i}"
            child_ids = []
            for child_idx in mapping:
                child_id = f"{book_key}_{'leaf' if depth == 0 else f'summary_{depth-1}'}_{child_idx}"
                child_ids.append(child_id)

            self.tree_structure[book_key]["children"][parent_id].extend(child_ids)
            for child_id in child_ids:
                self.tree_structure[book_key]["parents"][child_id].append(parent_id)
            update_mappings.append([parent_id, child_ids])
        
        self.dataset[book_key]["summary_layers"][depth] = summary_chunks
        self.dataset[book_key]["mapping_layers"][depth] = update_mappings

    def get_node_info(self, book_id, node_id):
        book_key = self.available_book_ids[book_id]
        return self.tree_structure[book_key]["nodes"].get(node_id, None)
    
    def get_node_parents(self, book_id, node_id):
        book_key = self.available_book_ids[book_id]
        return self.tree_structure[book_key]["parents"].get(node_id, None)
    
    def get_node_children(self, book_id, node_id):
        book_key = self.available_book_ids[book_id]
        return self.tree_structure[book_key]["children"].get(node_id, None)

    def update_book_summary(self, book_id, depth, summaries, mappings):
        book_key = self.available_book_ids[book_id]
        if "summary_layers" not in self.dataset[book_key]:
            self.dataset[book_key]["summary_layers"] = {}
            self.dataset[book_key]["mapping_layers"] = {}
        
        summary_chunks = []
        for i, summary in enumerate(summaries):
            summary_id = f"{book_key}_summary_{depth}_{i}"
            summary_data = {
                "id": summary_id,
                "text": summary,
                "type": "summary",
            }
            summary_chunks.append(summary_data)

            # update the tree structure.
            self.tree_structure[book_key]["nodes"][summary_id] = {
                "text": summary,
                "level": f"depth_{depth}",
                "type": "summary",
            }
            self.tree_structure[book_key]["children"][summary_id] = []
            self.tree_structure[book_key]["parents"][summary_id] = []

        #update the mapping relations.
        update_mappings = []
        for i, mapping in enumerate(mappings):
            parent_id = f"{book_key}_summary_{depth}_{i}"
            child_ids = []
            for child_idx in mapping:
                if depth == 0:
                # which means the children is original chunks.
                    child_id = f"{book_key}_leaf_{child_idx}"
                else:
                # which means the children is summary.
                    child_id = f"{book_key}_summary_{depth-1}_{child_idx}"
                child_ids.append(child_id)

            # update the parent and children.
            self.tree_structure[book_key]["children"][parent_id].extend(child_ids)
            for child_id in child_ids:
                self.tree_structure[book_key]["parents"][child_id].append(parent_id)
            update_mappings.append([parent_id, child_ids])
        
        self.dataset[book_key]["summary_layers"][depth] = summary_chunks
        self.dataset[book_key]["mapping_layers"][depth] = update_mappings

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
    
    def load_summary(self, summary_folder:str, extraction_folder:str):
        """load the summary and the node chunk mapping."""
        for file in os.listdir(summary_folder):
            with open(os.path.join(summary_folder, file), "r") as infile:
                book_summary = json.loads(infile.read())
                book_id = file.split('.')[0].split('_')[1]
                self.dataset[book_id]["summary_layers"] = book_summary["summary_layers"]
                self.dataset[book_id]["mapping_layers"] = book_summary["mapping_layers"]

                # Initialize tree structure if not exists
            if book_id not in self.tree_structure:
                self.tree_structure[book_id] = {
                    "nodes": {},
                    "children": {},
                    "parents": {}
                }
            
            # Update tree structure for each summary layer
            for depth, summaries in book_summary["summary_layers"].items():
                depth = int(depth)
                for i, summary in enumerate(summaries):
                    summary_id = f"{book_id}_summary_{depth}_{i}"
                    
                    # Add node info
                    self.tree_structure[book_id]["nodes"][summary_id] = {
                        "text": summary["text"],
                        "level": f"depth_{depth}",
                        "type": "summary"
                    }
                    
                    # Initialize parent/child lists
                    if summary_id not in self.tree_structure[book_id]["children"]:
                        self.tree_structure[book_id]["children"][summary_id] = []
                    if summary_id not in self.tree_structure[book_id]["parents"]:
                        self.tree_structure[book_id]["parents"][summary_id] = []
            
            # Update parent-child relationships
            for depth, mappings in book_summary["mapping_layers"].items():
                depth = int(depth)
                for i, mapping in enumerate(mappings):
                    parent_id = f"{book_id}_summary_{depth}_{i}"
                    for child_idx in mapping[1]:  # mapping[1] contains child indices
                        if depth == 1:  # connecting to leaf nodes
                            child_id = f"{book_id}_leaf_{child_idx}"
                        else:  # connecting to other summary nodes
                            child_id = f"{book_id}_summary_{depth-1}_{child_idx}"
                        
                        # Update bidirectional relationships
                        self.tree_structure[book_id]["children"][parent_id].append(child_id)
                        if child_id not in self.tree_structure[book_id]["parents"]:
                            self.tree_structure[book_id]["parents"][child_id] = []
                        self.tree_structure[book_id]["parents"][child_id].append(parent_id)

        #loading the index.
        for file in os.listdir(extraction_folder):
            with open(os.path.join(extraction_folder, file), "r") as infile:
                book_id = file.split('_')[0]
                node_chunk_mapping = json.loads(infile.read())
                node_chunk_mapping["global_nouns"] = set(node_chunk_mapping["global_nouns"])
                node_chunk_mapping["chunk_to_nouns"] = {k: set(v) for k, v in node_chunk_mapping["chunk_to_nouns"].items()}
                node_chunk_mapping["noun_to_chunks"] = {k: set(v) for k, v in node_chunk_mapping["noun_to_chunks"].items()}
                node_chunk_mapping["noun_pairs"] = {
                    tuple(k.split("<|COMMA|>")): v
                    for k, v in node_chunk_mapping["noun_pairs"].items()
                }
                self._index[book_id] = node_chunk_mapping

    def update_index(self, book_id, index_dict):
        self._index[book_id] = index_dict

    def save_index(self, book_id, folder:str):
        assert book_id in self._index, "Book id not found in the index."
        with open(os.path.join(folder, f"{book_id}_index.json"), "w") as outfile:
            saving = self._index[book_id]
            saving["global_nouns"] = list(saving["global_nouns"])
            saving["noun_to_chunks"] = {k: list(v) for k, v in saving["noun_to_chunks"].items()}
            saving["chunk_to_nouns"] = {k: list(v) for k, v in saving["chunk_to_nouns"].items()}
            saving["noun_pairs"] = {f"{k[0]}<|COMMA|>{k[1]}":v for k, v in saving["noun_pairs"].items()}
            json.dump(saving, outfile, indent=4)

    def save_summary(self, book_id, folder:str):
        book_data = {
            "tree_structure": self.tree_structure[book_id],
            "summary_layers": self.dataset[book_id]["summary_layers"],
        }

        with open(os.path.join(folder, f"{book_id}_summary.json"), "w") as outfile:
            json.dump(book_data, outfile, indent=4)


class NovelQALoader(AbstractDataLoader):
    """
        self.dataset: {
            book_id: {
                "book": str,
                "book_chunks": List[{
                    "id": str,
                    "text": str,
                    "type": str
                }],
                "qa": List[Dict],
                "summary_layers": {
                    depth: List[Dict]
                },
                "mapping_layers": {
                    depth: List[List]
                }
            }
        }

        self.tree_structure: {
            book_id: {
                "nodes": {
                    node_id: {
                        "text": str,
                        "level": str,
                        "type": str,
                    }
                },
                "children": {
                    node_id: List[str]
                },
                "parents": {
                    node_id: List[str]
                }
            }
        }

    id naming rule:
        - leaf node: "{book_id}_leaf_{chunk_idx}"
        - summary node: "{book_id}_depth{depth}_{summary_idx}"
    """
    def __init__(self, 
                 saving_folder:str,
                 tokenizer:AutoTokenizer = None,
                 chunk_size:int = 1200,
                 overlap:int = 100,
                 load_summary_index:bool = False,
                 ) -> None:
        super().__init__()
        self.parent_folder = saving_folder
        self.qapath = os.path.join(saving_folder, "Data")
        self.docpath = os.path.join(saving_folder, "Books")
        self.tree_structure = {}
        self.dataset = self.build_dataset(datapath=self.qapath, bookpath=self.docpath)
        self.available_book_ids = list(self.dataset.keys())
        self.available_book_ids.sort()
        self.tokenizer = tokenizer
        self.dataset = self._chunk_book(self.tokenizer, chunk_size=chunk_size, overlap=overlap)
        self._index = {key: chunk_index() for key in self.available_book_ids}

        if load_summary_index:
            self.load_summary(summary_folder=f"{self.parent_folder}/Summary/0107", extraction_folder=f"{self.parent_folder}/Index")

       
    def build_dataset(self):
        '''Build the dataset.'''
        dataset = {}
        for root, dirs, files in os.walk(self.docpath, topdown=True):
            for directory in dirs:   # Copyright Protected and Public Domain
                for filename in os.listdir(os.path.join(self.docpath, directory)):
                    with open(os.path.join(self.docpath, directory, filename), "r") as infile:
                        book_id = filename.split('.')[0]
                        dataset[book_id] = {}
                        dataset[book_id]["book"] = infile.read()
        for root, dirs, files in os.walk(self.qapath, topdown=True):
            for directory in dirs:   # Copyright Protected and Public Domain
                for filename in os.listdir(os.path.join(self.qapath, directory)):
                    with open(os.path.join(self.qapath, directory, filename), "r") as infile:
                        qa_id = filename.split('.')[0]
                        dataset[qa_id]["qa"] = json.loads(infile.read())
        return dataset
    
    def save_res(self, ava_book_id, ans, save_folder:str):
        book_data = self.dataset[self.available_book_ids[ava_book_id]]
        book_data["answer"] = ans
        self.dataset[self.available_book_ids[ava_book_id]] = book_data
        with open(os.path.join(save_folder, f"{self.available_book_ids[ava_book_id]}.json"), "w") as outfile:
            json.dump(book_data["answer"], outfile, indent=4)
        return self.dataset
    
    def cal_metrics(self):
        '''
        Acc
        '''
        question_count = 0
        correct_count = 0
        for book_id in self.available_book_ids:
            book_data = self.dataset[book_id]
            for i, qa in enumerate(book_data["qa"]):
                question_count += 1
                if qa["Gold"] == book_data["answer"][i]:
                    correct_count += 1
        return correct_count / question_count



class NarrativeQALoader(AbstractDataLoader):
    """
        self.dataset: {
            book_id: {
                "book": str,
                "book_chunks": List[{
                    "id": str,
                    "text": str,
                    "type": str
                }],
                "qa": List[Dict],
                "summary_layers": {
                    depth: List[Dict]
                },
                "mapping_layers": {
                    depth: List[List]
                }
            }
        }

        self.tree_structure: {
            book_id: {
                "nodes": {
                    node_id: {
                        "text": str,
                        "level": str,
                        "type": str,
                    }
                },
                "children": {
                    node_id: List[str]
                },
                "parents": {
                    node_id: List[str]
                }
            }
        }

    id naming rule:
        - leaf node: "{book_id}_leaf_{chunk_idx}"
        - summary node: "{book_id}_depth{depth}_{summary_idx}"
    """
    def __init__(self,
                 tokenizer:AutoTokenizer = None,
                 chunk_size:int = 1200,
                 overlap:int = 100,
                 load_summary_index:bool = False,
                 saving_folder:str = "NarrativeQA"
                 ) -> None:
        super().__init__()
        # we only use the test set for evaluation.
        self.parent_folder = saving_folder
        origin_dataloader = load_dataset("narrativeqa")["test"]
        self.dataset, self.available_book_ids = self.build_dataset(origin_dataloader)
        self.tree_structure = {}
        self.dataset = self._chunk_book(tokenizer, chunk_size=chunk_size, overlap=overlap)
        self._index = {key: chunk_index() for key in self.available_book_ids}
        if load_summary_index:
            self.load_summary(summary_folder=f"{self.parent_folder}/Summary/0107", extraction_folder=f"{self.parent_folder}/Index")
        

    
    def build_dataset(self, dataset):
        '''format the dataset into a unified format.'''
        new_dataset = {}
        available_book_ids = set()
        # print(dataset[0].keys())
        for item in tqdm(dataset, desc = "Building Dataset"):
            item_data = {}
            item_data["book_id"] = item["document"]["id"]
            if item_data["book_id"] not in available_book_ids:
                available_book_ids.add(item_data["book_id"])
                item_data["book"] = item["document"]["text"]
                item_data["book_chunks"] = []
                item_data["summary_provided"] = item["document"]["summary"]["text"]
                item_data["qa"] = {item["question"]["text"]: item["answers"]}
                new_dataset[item_data["book_id"]] = item_data
                available_book_ids.add(item_data["book_id"])
            else:
                item_data["qa"] = {item["question"]["text"]: item["answers"]}
                new_dataset[item_data["book_id"]]["qa"].update(item_data["qa"])
        available_book_ids = list(available_book_ids)
        available_book_ids.sort()
        # print(len(available_book_ids))
        return new_dataset, available_book_ids



if __name__ == "__main__":
    dataloader = NovelQALoader(docpath="NovelQA/Books", qapath="NovelQA/Data/PublicDomain")
    print(len(dataloader))
    print(dataloader[0]["book_chunks"][0])