import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from build_tree import build_tree
from extract_graph import extract_graph, load_nlp

def clean_cuda_memory(device_id):
    """清理指定GPU设备的缓存"""
    if torch.cuda.is_available():
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()

def build_tree_task(args):
    llm_path, llm_device, text, cache_folder, tokenizer_name, length, overlap, merge_num = args
    try:
        device_id = int(llm_device.split(':')[1]) if ':' in llm_device else 0
        
        # Load model and tokenizer in subprocess
        llm = AutoModel.from_pretrained(llm_path)
        if "Qwen2Model" in str(type(llm)):
            from transformers import Qwen2ForCausalLM
            llm = Qwen2ForCausalLM.from_pretrained(llm_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        llm_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer, device=llm_device)
        
        # Process
        result = build_tree(text, llm_pipeline, cache_folder, tokenizer, length, overlap, merge_num)
        return result
    finally:
        # Clean up
        del llm_pipeline
        del llm
        del tokenizer
        clean_cuda_memory(device_id)

def extract_graph_task(args):
    text, cache_folder = args
    try:
        # Load NLP model in subprocess
        nlp = load_nlp()
        result = extract_graph(text, cache_folder, nlp)
        return result
    finally:
        # Clean up
        del nlp 