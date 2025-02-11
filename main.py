import multiprocessing as mp
from build_tree import build_tree
from extract_graph import extract_graph

def parallel_build_extract(text, llm, cache_path, tokenizer, length, overlap, merge_num):

    build_tree_process = mp.Process(target=build_tree, args=(text, llm, cache_path, tokenizer, length, overlap, merge_num))

    extract_process = mp.Process(target=extract_graph, args=(text, cache_path))

    build_tree_process.start()
    extract_process.start()

    build_tree_process.join()
    extract_process.join()

def main():
    # parse the arguments.

    # load the dataset.

    # start calculate the time.

    # chunk the dataset.

    # for data_piece in dataset:
        # parallel_build_extract(data_piece) - save the cache.
        # answer the question.
        # save the answer.

    # end for loop.

    # end calculate the time.

    # evaluate the answer.

    # save the result.

    # end.

    pass

if __name__ == "__main__":
    main()