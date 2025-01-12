'''
Simply send the full text to the LLM.
if the text is too long, chunk it at the max token limit.
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yb_dataloader import NovelQALoader, NarrativeQALoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from baseline_prompts import full_text_prompt
import argparse
import logging
from tqdm import tqdm
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("full text baseline")

def get_full_text_input(text: str, question:str, tokenizer: AutoTokenizer) -> str:
    formatted_prompt = full_text_prompt.format(question=question)
    
    prompt_tokens = len(tokenizer(formatted_prompt, return_length=True)['input_ids'])
    
    max_length = tokenizer.model_max_length
    
    # leave 100 tokens for the answer
    remaining_length = max_length - prompt_tokens - 100
    
    if len(tokenizer(text, return_length=True)['input_ids']) > remaining_length:
        encoded_text = tokenizer.encode(text, truncation=True, max_length=remaining_length)
        text = tokenizer.decode(encoded_text, skip_special_tokens=True)
    
    return formatted_prompt + text + "\n Answer:"

def full_text_answer(text: str, question:str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> str:
    input_text = get_full_text_input(text, question, tokenizer)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=tokenizer.model_max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_text):]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/root/shared_planing/LLM_model/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="NovelQA")
    parser.add_argument("--bits", type=int, default=16, choices=[4, 8], help="Quantization bits (4 or 8)")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="Comma-separated GPU IDs to use")
    args = parser.parse_args()

    # 设置要使用的GPU
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    device_map = {i: f"cuda:{id}" for i, id in enumerate(gpu_ids)}
    
    # 量化配置
    if args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    elif args.bits == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        logger.warning(f"Unsupported bits value: {args.bits}. Using full precision.")
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",  # 或者使用 "sequential"
        max_memory={i: "78GiB" for i in gpu_ids},  # 为每个GPU分配内存
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model.eval()

    logger.info(f"Loading {args.dataset} dataset...")
    if args.dataset == "NovelQA":
        loader = NovelQALoader(docpath="../NovelQA/Books/PublicDomain", qapath="../NovelQA/Data/PublicDomain", tokenizer=tokenizer)
    elif args.dataset == "NarrativeQA":
        loader = NarrativeQALoader(docpath="../NarrativeQA/Books/PublicDomain", qapath="../NarrativeQA/Data/PublicDomain", tokenizer=tokenizer)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    logger.info(f"Running full text baseline on {args.dataset} dataset...")

    for bid in tqdm(loader.available_books):
        book = loader.dataset[bid]["book"]
        questions = loader.dataset[bid]["qa"]
        for question in questions:
            question_text = questions[question]["Question"]
            question_options = questions[question]["Options"]
            options_text = "\n".join([f"{k}: {v}" for k, v in question_options.items()])
            input_question = f"{question_text}\n{options_text}"
            answer = full_text_answer(book, input_question, model, tokenizer)
            print(answer)
            break
        break

if __name__ == "__main__":
    main()