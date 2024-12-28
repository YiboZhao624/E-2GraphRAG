'''
Simply send the full text to the LLM.
if the text is too long, chunk it at the max token limit.
'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from baseline_prompts import full_text_prompt
from ..yb_dataloader import NovelQALoader, NarrativeQALoader
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("full text baseline")

def get_full_text_input(text: str, question:str, tokenizer: AutoTokenizer) -> str:
    formatted_prompt = full_text_prompt.format(question=question)
    
    prompt_tokens = tokenizer(formatted_prompt, return_length=True)['length']
    
    max_length = tokenizer.model_max_length
    
    # leave 100 tokens for the answer
    remaining_length = max_length - prompt_tokens - 100
    
    if tokenizer(text, return_length=True)['length'] > remaining_length:
        encoded_text = tokenizer.encode(text, truncation=True, max_length=remaining_length)
        text = tokenizer.decode(encoded_text, skip_special_tokens=True)
    
    return formatted_prompt + text

def full_text_answer(text: str, question:str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> str:
    input_text = get_full_text_input(text, question, tokenizer)
    return model.generate(input_text, max_length=tokenizer.model_max_length, num_return_sequences=1)[0][len(input_text):]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama3-8b-8192")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="NarritiveQA")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)


    if args.dataset == "NovelQA":
        loader = NovelQALoader(docpath="../NovelQA/Books", qapath="../NovelQA/Data", tokenizer=tokenizer)
    elif args.dataset == "NarrativeQA":
        loader = NarrativeQALoader()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    for bid in loader.available_books:
        book = loader.dataset[bid]["book"]
        question = loader.dataset[bid]["question"]
        answer = full_text_answer(book, question, model, tokenizer)
        print(answer)
        break

if __name__ == "__main__":
    main()