from datasets import load_dataset
import os
import json
from tqdm import tqdm

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



class NarrativeQALoader:
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
    def __init__(self):
        origin_dataset = load_dataset("narrativeqa")["test"]
        self.dataset = self._format_dataset(origin_dataset)
        self.available_ids = list(self.dataset.keys())

    def _format_dataset(self, dataset):
        formatted_dataset = {}
        for item in tqdm(dataset, desc = "Formatting Dataset"):
            book_id = item["document"]["id"]
            if book_id not in formatted_dataset:
                formatted_dataset[book_id] = {}
                formatted_dataset[book_id]["book"] = item["document"]["text"]
                formatted_dataset[book_id]["qa"] = []
            formatted_dataset[book_id]["qa"].append({
                "question": item["question"]["text"],
                "answer": [item["answers"][i]["text"] for i in range(len(item["answers"]))]
            })
        return formatted_dataset
    
    def __getitem__(self, index):
        return self.dataset[self.available_ids[index]]


class test_loader:
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
        dataset = {0:{}}
        doc_path = os.path.join(self.docpath, "PublicDomain","B00.txt")
        qa_path = os.path.join(self.qapath, "PublicDomain", "B00.json")
        with open(doc_path, "r") as infile:
            dataset[0]["book"] = infile.read()
        with open(qa_path, "r") as infile:
            dataset[0]["qa"] = json.loads(infile.read())
        
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
                "answer": answer
            })
        return formatted_qa

    def __getitem__(self, index):
        to_return = {}
        to_return["book"] = self.dataset[index]["book"]
        to_return["qa"] = self._format_qa(self.dataset[index]["qa"])
        return to_return

    def __len__(self):
        return 1




if __name__ == "__main__":
    loader = NarrativeQALoader()
    print("narrativeqa")
    loader = NovelQALoader("NovelQA")
    print(loader[0])
    loader = test_loader("NovelQA")
    print(loader[0])    