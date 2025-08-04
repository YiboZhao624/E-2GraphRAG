from datasets import load_dataset
import os
import json
import logging
from tqdm import tqdm

# Get logger for this module
logger = logging.getLogger(__name__)

class NovelQALoader:
    '''
    data will be returned as a dictionary with :
    {
        "book": str,
        "qa": list of dictionaries with keys:
            "id": str,
            "question": str,
            "answer": str
    }
    question is well formatted.
    '''
    def __init__(self, path):
        self.parent_folder = path
        self.qapath = os.path.join(path, "Data")
        self.docpath = os.path.join(path, "Books")
        self.dataset = self._initialize_dataset()
        self.available_ids = list(self.dataset.keys())
    
    def _initialize_dataset(self):
        dataset = {}
        for root, dirs, files in os.walk(self.docpath):
            for directory in dirs:
                # copyright protected and public domain
                for filename in os.listdir(os.path.join(self.docpath, directory)):
                    with open(os.path.join(self.docpath, directory, filename), "r") as infile:
                        book_id = int(filename.split('.')[0][1:])
                        dataset[book_id] = {}
                        dataset[book_id]["book"] = infile.read()
        for root, dirs, files in os.walk(self.qapath):
            for directory in dirs:
                for filename in os.listdir(os.path.join(self.qapath, directory)):
                    with open(os.path.join(self.qapath, directory, filename), "r") as infile:
                        qa_id = int(filename.split('.')[0][1:])
                        dataset[qa_id]["qa"] = json.loads(infile.read())
        return dataset
    
    def _format_qa(self, qa_dict):
        formatted_qa = []
        for qa_id, qa in qa_dict.items():
            question = qa["Question"]
            options = qa["Options"]
            answer = qa["Gold"]
            question_text = question + "\n"
            for option, text in options.items():
                question_text += option + ". " + text
                if option != "D":
                    question_text += "\n"
            formatted_qa.append({
                "id": qa_id,
                "question": question_text,
                "answer": answer,
                "evidence": qa["Evidences"]
            })
        return formatted_qa

    def __getitem__(self, index):
        index = self.available_ids[index]
        to_return = {}
        to_return["book"] = self.dataset[index]["book"]
        to_return["qa"] = self._format_qa(self.dataset[index]["qa"])
        return to_return

    def __len__(self):
        return len(self.dataset)

class InfiniteQALoader:
    def __init__(self, path = "./InfiniteBench/data/longbook_qa_eng.jsonl"):
        self.dataset, self.available_ids = self._initialize_dataset(path)

    def _initialize_dataset(self, path):
        dataset = {}
        context_id = -1
        prev_context = ""
        
        with open(path, "r", encoding='utf-8') as infile:  # Add encoding='utf-8'
            for line in infile:
                try:
                    data = json.loads(line.strip())  # Add strip() to remove whitespace/newlines
                    context = data["context"]
                    question = data["input"]
                    answer = data["answer"]
                    
                    if context != prev_context:
                        context_id += 1
                        prev_context = context
                        dataset[context_id] = {
                            "book": context,
                            "qa": []
                        }
                    
                    dataset[context_id]["qa"].append({
                        "question": question,
                        "answer": answer
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"Warning: Skipping invalid JSON line: {e}")
                    continue
                    
        return dataset, list(dataset.keys())

    def __getitem__(self, index):
        return self.dataset[self.available_ids[index]]

    def __len__(self):
        return len(self.available_ids)


class InfiniteChoiceLoader:
    def __init__(self, path = "./InfiniteBench/data/longbook_choice_eng.jsonl"):
        self.dataset, self.available_ids = self._initialize_dataset(path)

    def _initialize_dataset(self, path):
        dataset = {}
        with open(path, "r") as infile:
            prev_context = ""
            context_id = -1
            for line in infile:
                data = json.loads(line)
                context = data["context"]
                question = data["input"]
                answer = data["answer"]
                if context != prev_context:
                    context_id += 1
                    prev_context = context
                    dataset[context_id] = {}
                    dataset[context_id]["book"] = context
                    dataset[context_id]["qa"] = []
                qa_dict = {
                    "question": question,
                    "answer": answer,
                    "options": data["options"]
                }
                dataset[context_id]["qa"].append(self._format_qa(qa_dict))
        return dataset, list(dataset.keys())

    def _format_qa(self, qa_dict):
        res = {}
        formatted_question = ""
        formatted_question += qa_dict["question"] + "\n"
        option_name = ["A", "B", "C", "D"]
        for i in range(len(qa_dict["options"])):
            formatted_question += option_name[i] + ". " + qa_dict["options"][i] + "\n"
            if qa_dict["answer"][0] == qa_dict["options"][i]:
                res["answer"] = option_name[i]
        res["question"] = formatted_question
        return res
    
    def __getitem__(self, index):
        data = self.dataset[self.available_ids[index]]
        return data

    def __len__(self):
        return len(self.available_ids)


if __name__ == "__main__":
    loader = NovelQALoader("NovelQA")
    print(loader[0]["qa"][0])    
    print(loader[0]["qa"][0]["question"].split("\n")[0])
    loader = InfiniteChoiceLoader()
    print(loader[0].keys())
    print(type(loader[0]["book"]))
    print(loader[0]["qa"])
    loader = InfiniteQALoader()
    print(loader[0].keys())
    print(type(loader[0]["book"]))
    print(loader[0]["qa"])
    print(len(loader))