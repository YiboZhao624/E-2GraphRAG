## 1. 建图:

1.1 分chunk

不同大小的chunk是否要做一个消融-没空做了。。

1.2 对chunk抽三元组/实体

直接抽名词和共现次数就好。

1.3 实体去重

可以直接复用之前的BERT Embedding聚类的方法，将同一个实体的名称都标准化。

1.4 实体-chunk mapping

这里需要考虑LLM从query里抽的实体是否够标准？或者这一步将同一个实体的所有名字都保留下来，即一个实体有多个名字，只要chunk中出现了一个名字，就将所有别名都作为这一个小chunk的key。

## 2. 建树

2.1 chunk cluster

这里是否可以加一点，比如有的topic可能是三个chunk讲的，有的是两个chunk，我们如果直接固定了summary对应的chunk数目，可能会导致同一个话题分到了两个父节点，我们自顶向下的查询可能就不够准，所以加一步cluster可能更灵活、更好。

也可以直接用超参，无所谓，我们就是快！

2.2 chunk summary

直接调用大模型做summary

## 3. 查询

3.1 对Query实体识别

直接用大模型

3.2 叶子节点查询

查询树的叶子节点的key

if mapped：从叶子节点出发查询

if not mapped：从根节点出发查询

3.3 过滤+排序

计算查询到的树节点和query的相似度--这里直接用相似度足够好吗？好像也没想到新的方法，然后得到排序结果

## 4. 评估

NovelQA + NarrativeQA + Summarize case study

## 5. 数据集

https://openrag.notion.site/9f4033e589e14a2f9e4ee8d2eb69b7ef?v=7750086caa7c40109d34a84a754b7de6

**Long Form Long Doc QA:**

**narrative qa**: https://github.com/google-deepmind/narrativeqa?tab=readme-ov-file

ELI5: https://github.com/facebookresearch/ELI5/tree/main

QMSum: https://github.com/Yale-LILY/QMSum

Loong: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/Loong

FindSum: https://drive.google.com/drive/folders/1KVJAKWO49Iahgsv4lipn49pM-ojXL6yM

**NovelQA**: https://github.com/NovelQA/novelqa.github.io --> Local, long document.
