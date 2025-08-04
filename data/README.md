# How to obtain the data

## NovelQA

NovelQA is created by the authors of paper [NovelQA: Benchmarking Question Answering on Documents Exceeding 200K Tokens](https://arxiv.org/abs/2403.12766).

Due to some books included in the NovelQA is not publicly available, we cannot provide the whole dataset. If you want to get the whole dataset, please contact the authors of NovelQA.

For the available part of data, you can obtain it folloing the instructions in the directory `./data/NovelQA`:

```bash
pip install huggingface_hub
huggingface-cli login
git clone https://huggingface.co/datasets/NovelQA/NovelQA
```

## InfiniteBench

InfiniteBench is created by the authors of paper [\inftyBench: Extending Long Context Evaluation Beyond 100K Tokens](https://arxiv.org/abs/2402.13718).

For simplicity, you can download the file `longbook_choice_eng.jsonl` and `longbook_qa_eng.jsonl` from the link [InfiniteBench](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/tree/main) and put them in the directory `./data/InfiniteBench`. We also provide the downloading script `download.sh` originally from the InfinteBench repo.
