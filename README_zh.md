# E²GraphRAG：高效且有效的图结构增强式检索生成框架

E²GraphRAG 是一个轻量级、模块化的框架，旨在提升基于图的增强式检索生成（Retrieval-Augmented Generation, RAG）在 **效率** 和 **效果** 两方面的表现。该框架通过结构化图推理，简化了从文档解析到答案生成的整个流程。

## 📁 项目结构

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


## 📦 数据集

我们使用了以下两个数据集：

- [📚 NovelQA](https://huggingface.co/datasets/NovelQA/NovelQA)  
  部分开源，若需获取完整数据集，请联系原作者申请。
  
- [🔁 InfiniteBench](https://github.com/OpenBMB/InfiniteBench)  
  完全开源，公开可用。

你可以在 `./data/README.md` 中找到获取数据的具体说明。

> **注意：** 获取数据后，请在初始化 `Dataloader` 类时指定数据路径。

## 🚀 快速开始

### 1. 安装依赖

请先安装项目所需的 Python 库：

```bash
pip install -r requirements.txt
```

### 2. 运行主流程

整个流程包括构建文档树、提取图结构并生成答案，主程序入口为 `main.py`。

运行步骤如下：

> Step 1：准备配置文件
> 使用 YAML 格式的配置文件来定义关键参数。

> 👉 示例路径：./configs/example_config.yaml

Step 2：运行主程序
> ```
>  python main.py --config <配置文件路径>
> ```

### 📬 联系我们 & 引用

如果您在研究中使用了本代码，或者觉得本项目对您有帮助，欢迎引用我们的工作。如有问题，或需获取 NovelQA 数据集，请联系原作者。
