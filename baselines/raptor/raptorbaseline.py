from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from prompts import SUMMARY_PROMPT, QA_PROMPT_RAPTOR
import argparse
from yb_dataloader import NovelQALoader, NarrativeQALoader
import os
import torch
import numpy as np

class CustomSummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name:str, tokenizer:AutoTokenizer, device:str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    def summarize(self, text:str, max_tokens:int=1024) -> str:
        prompt = SUMMARY_PROMPT.format(text=text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]


class CustomQAModel(BaseQAModel):
    def __init__(self, model_name:str, tokenizer:AutoTokenizer, device:str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    def answer_question(self, context, question):
        if len(question.split("\n"))>2:
            multiple_choice = True
        else:
            multiple_choice = False
        if not multiple_choice: 
            prompt = QA_PROMPT_RAPTOR.format(question=question, evidence=context)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=50)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        else:
            prompt = QA_PROMPT_RAPTOR.format(question=question, evidence=context)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.model.device)
            outputs = self.model(input_ids = inputs).logits[0, -1]
            probs = torch.nn.functional.softmax(
                torch.tensor([
                        outputs[self.tokenizer("A").input_ids[-1]],
                        outputs[self.tokenizer("B").input_ids[-1]],
                        outputs[self.tokenizer("C").input_ids[-1]],
                        outputs[self.tokenizer("D").input_ids[-1]],
                    ]).float(),
                    dim=0,
                ).detach().cpu().numpy()
            answer = ["A", "B", "C", "D"][np.argmax(probs)]
            return answer

class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name:str, tokenizer:AutoTokenizer, device:str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    def create_embedding(self, text:str) -> list[float]:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()[0]
    


# tokenizer = AutoTokenizer.from_pretrained("/root/shared_planing/LLM_model/Qwen2.5-14B-Instruct")

# custom_summarizer = CustomSummarizationModel(model_name="/root/shared_planing/LLM_model/Qwen2.5-14B-Instruct", tokenizer=tokenizer, device="cuda:4")

# custom_qa_model = CustomQAModel(model_name="/root/shared_planing/LLM_model/Qwen2.5-14B-Instruct", tokenizer=tokenizer, device="cuda:6")

# custom_embedding_model = CustomEmbeddingModel(model_name="/root/shared_planing/LLM_model/Qwen2.5-14B-Instruct", tokenizer=tokenizer, device="cuda:6")

# custom_config = RetrievalAugmentationConfig(
#     summarization=custom_summarizer,
#     qa_model=custom_qa_model,
#     embedding_model=custom_embedding_model,
#     tb_max_tokens=1200,
#     tb_num_layers=5,
#     tb_tokenizer=tokenizer,
# )




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--embedding_model_name", type=str, required=True)
    parser.add_argument("--summary_device", type=str, required=True)
    parser.add_argument("--qa_device", type=str, required=True)
    parser.add_argument("--embedding_device", type=str, required=True)
    args = parser.parse_args()

    dataset = args.dataset
    output_folder = args.output_folder
    llm_name = args.llm_name
    embedding_model_name = args.embedding_model_name
    summary_device = args.summary_device
    qa_device = args.qa_device
    embedding_device = args.embedding_device

    tokenizer = AutoTokenizer.from_pretrained(llm_name)

    custom_summarizer = CustomSummarizationModel(model_name=llm_name, tokenizer=tokenizer, device=summary_device)

    custom_qa_model = CustomQAModel(model_name=llm_name, tokenizer=tokenizer, device=qa_device)

    custom_embedding_model = CustomEmbeddingModel(model_name=embedding_model_name, tokenizer=tokenizer, device=embedding_device)

    custom_config = RetrievalAugmentationConfig(
        summarization_model=custom_summarizer,
        qa_model=custom_qa_model,
        embedding_model=custom_embedding_model,
        tb_tokenizer=tokenizer,
        tb_max_tokens=1200,
        tb_num_layers=5,
    )

    

    if dataset == "LihuaWorld":
        raise NotImplementedError("LihuaWorld is not implemented yet.")
    elif dataset == "NarrativeQA":
        dataloader = NarrativeQALoader(tokenizer=tokenizer, chunk_size=1200, overlap=100, load_summary_index=False, saving_folder="/root/projects/lightTAG/NarrativeQA")
    elif dataset == "NovelQA":
        dataloader = NovelQALoader(tokenizer=tokenizer, chunk_size=1200, overlap=100, load_summary_index=False, saving_folder="/root/projects/lightTAG/NovelQA")
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")


    for book_id in dataloader.available_book_ids:
        RA = RetrievalAugmentation(config = custom_config)
        text = dataloader.dataset[book_id]["book"]
        RA.add_documents(text)
        res_log = os.path.join(output_folder, f"{book_id}.txt")
        with open(res_log, "w") as outfile:
            for question in dataloader.dataset[book_id]["qa"]:
                answer = RA.answer_question(question)
                outfile.write(f"{question}\n{answer}\n\n")

if __name__ == "__main__":
    main()