import multiprocessing as mp
from build_tree import build_tree
from noun_extracting import extract_nouns

def parallel_build_extract(text, llm, cache_path, tokenizer, length, overlap, merge_num):

    build_tree_process = mp.Process(target=build_tree, args=(text, llm, cache_path, tokenizer, length, overlap, merge_num))

    extract_process = mp.Process(target=extract_nouns, args=(text))

    build_tree_process.start()
    extract_process.start()

    build_tree_process.join()
    extract_process.join()

