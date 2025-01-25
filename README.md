## 文档树

1. sequential，直接聚合做summarize
2. 索引键：利用NER从中提取实体，同一个句子中的实体算一次共现，有一个边关系

## 建图

1. 实体共现作为知识图谱
2. 查询时从图谱中找实体之间的最短路径，实现多跳的推理
3. 将关键路径同时送入到模型中 prompt类似于“A、B、C”对这个问题很重要
4. 建图和文档树构建是并行的

## 查询

1. 对query做NER，提取实体
2. 到索引键中进行map，如果有map到的，那么对结果去交集，同时去除掉其中没有任何map到的实体
3. 如果全都没有map到，或者没有实体，那么就从顶开始查询
4. 从顶到下设置不同的similarity阈值，从而避免过于**武断**的剪枝操作。

## 评估指标

dataset  RAPTOR  GraphRAG  LightRAG  Ours

NovelQA - Acc., Build Time, Query Time, Token used.

NarrativeQA - BLEU, Build Time, Query Time, Token used.

## 5. 数据集

https://openrag.notion.site/9f4033e589e14a2f9e4ee8d2eb69b7ef?v=7750086caa7c40109d34a84a754b7de6

**Long Form Long Doc QA:**

**narrative qa**: https://github.com/google-deepmind/narrativeqa?tab=readme-ov-file

ELI5: https://github.com/facebookresearch/ELI5/tree/main

QMSum: https://github.com/Yale-LILY/QMSum

Loong: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/Loong

FindSum: https://drive.google.com/drive/folders/1KVJAKWO49Iahgsv4lipn49pM-ojXL6yM

* [X]  **NovelQA**: https://github.com/NovelQA/novelqa.github.io --> Local, long document.

## Developing Roadmap:

1. refactor yb_dataloader
2. summarize Narrative QA
3. query function.
4. run query
5. run baselines

## Metrics

1. Acc
2. Retrieving Time（Ratio）
3. Building Time（Ratio）
4. Case Study

## Backbone Model

1. Qwen2.5-14B-Instruct
2. Llama3.1-7B-Instruct

## 结果数据

预构建时间：

Ours Novel QA：100hrs 37min 48sec

Ours Narrative QA：

RAPTOR Novel QA：
