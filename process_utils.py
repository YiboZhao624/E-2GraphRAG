import os
import torch
import logging
from transformers import pipeline, AutoTokenizer, AutoModel
from build_tree import build_tree, load_cache_summary
from extract_graph import extract_graph, load_nlp, load_cache

# Get logger for this module
logger = logging.getLogger(__name__)

def clean_cuda_memory(device_id):
    """清理指定GPU设备的缓存"""
    if torch.cuda.is_available():
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()

def build_tree_task(args):
    llm_path, llm_device, text, cache_folder, tokenizer_name, length, overlap, merge_num, torch_dtype, language = args
    if os.path.exists(os.path.join(cache_folder, "tree.json")):
        return load_cache_summary(os.path.join(cache_folder, "tree.json")), -1
    try:
        device_id = int(llm_device.split(':')[1]) if ':' in llm_device else 0
        
        # Load model and tokenizer in subprocess
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        llm_pipeline = pipeline(
            "text-generation", 
            model=llm_path, 
            tokenizer=tokenizer, 
            device=llm_device, 
            torch_dtype=torch_dtype,
            max_new_tokens=1200
        )
        
        # Process
        result, time_cost = build_tree(text, llm_pipeline, cache_folder, tokenizer, length, overlap, merge_num, language)
        logger.info(f"build tree task result type: {type(result)}")
        logger.info(f"build tree task time cost: {time_cost}, -1 means load from cache.")
        return result, time_cost
    except Exception as e:
        logger.error(f"build tree task error: {e}")
        logger.error(f"{type(e).__name__}")
        logger.error(f"{e.args}")
        import traceback
        logger.error(f"build tree task error stack:\n{traceback.format_exc()}")
        raise e
    finally:
        # Clean up
        del llm_pipeline
        del tokenizer
        clean_cuda_memory(device_id)

def extract_graph_task(args):
    text, cache_folder, language, method, force_reextract = args
    if os.path.exists(os.path.join(cache_folder, f"graph_{method}.json")) and not force_reextract:
        return load_cache(cache_folder, method), -1
    try:
        # Load NLP model in subprocess
        nlp = load_nlp(language, method)
        logger.info("NLP loaded.")
        (result, index, count), time_cost = extract_graph(text, cache_folder, nlp, use_cache=not force_reextract, reextract=force_reextract)
        logger.info(f"extract graph task result type: {type(result)}")
        logger.info(f"extract graph task time cost: {time_cost}, -1 means load from cache.")
        return (result, index, count), time_cost
    finally:
        # Clean up
        del nlp 