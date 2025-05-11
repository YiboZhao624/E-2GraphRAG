# README

## 1. File Structure

```
.
├── README.md
├── requirements.txt
├── main.py
├── build_tree.py
├── dataloader.py
├── extract_graph.py
├── GlobalConfig.py
├── process_utils.py
├── prompt_dict.py
├── query.py
└── utils.py
```

## 2. Data

We leverage the data from [NovelQA](https://huggingface.co/datasets/NovelQA/NovelQA) and [InfiniteBench](https://github.com/OpenBMB/InfiniteBench). All data from InfiniteBench is available publicly, while NovelQA data is available upon sending an email to the authors. After obtaining the data, please input the data path when initiating the dataloader class.

## 3. Usage

We provide the main.py file to run the entire pipeline, including the tree building, graph extraction, and answer generation. You need to first create a config.yaml file to specify the parameters, and then run the following command:

```bash
python main.py --config <path_to_config_file>
```

the example config file is provided in the ./configs/example_config.yaml

## 4. Reproduce the Results

To ensure a better reproducibility, we provide the built tree and extracted graph in the ./cache folder. You can directly use them and `main_cacheready.py` to reproduce the results. Due to some books in the NovelQA dataset are not available publicly, we provide the results of the public domain books only.