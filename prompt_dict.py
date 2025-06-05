Prompts = {
    
    "summarize_details":\
"""You are a helpful assistant that summarizes the details of a novel. You will be given a part of a novel. You need to summarize given content. The summary should include the main characters, the main plot and some other details. You need to return the summary in a concise manner without any additional fictive information. The length of the summary should be about 1000 tokens. 
Here is the content:
Content: {content}
Now, please summarize the content.
Summary: """,


    "summarize_summary":\
"""You are a helpful assistant that further summarizes the summaries of a novel. You will be given a series of summaries of parts of a novel. You need to summarize the summaries in a concise manner. The length of the summary should be about 1000 tokens.
Here is the summaries:
Summary: {summary}
Now, please summarize the summary based on the question.
Summary: """,

    
    "QA_prompt_options":\
"""You are a helpful assistant, you are given a question, please answer the question based on the given evidences. The answer should be an option among "A", "B", "C", and "D" that supported by the given evidences and matches the question. You should not assume any information beyond the evidence. You should only output the option. The format of the Evidence is keyEntity1_keyEntity2: Related chunks, which means the related chunks contain the information of the relationship between keyEntity1 and keyEntity2, where keyEntity1 and keyEntity2 are the entities in the question.

Question: {question}
Evidence: {evidence}

Answer: """,

    
    "QA_prompt_answer":\
"""You are a helpful assistant, you are given a question, please answer the question based on the given evidences. The answer should be a short sentence that supported by the given edidences and matches the requirements of the question. You should not assume any information beyond the evidence. You should only output the answer. The format of the Evidence is keyEntity1_keyEntity2: Related chunks, which means the related chunks contain the information of the relationship between keyEntity1 and keyEntity2, where keyEntity1 and keyEntity2 are the entities in the question.

Question: {question}
Evidence: {evidence}

Answer: """,


    "QA_prompt_answer_zh":\
"""你是一位擅长回答问题的助手。你将收到一个问题和一些证据，请根据证据回答问题。答案应该是一个简短的句子，支持给定的证据，并符合问题的要求。不要假设任何证据之外的信息。你只需要输出答案。证据的格式是keyEntity1_keyEntity2: Related chunks，表示相关片段包含keyEntity1和keyEntity2之间的关系信息，其中keyEntity1和keyEntity2是问题中的实体。

问题：{question}
证据：{evidence}

答案：""",

    "summarize_details_zh":\
"""你是一位擅长总结文本内容的助手。你将收到一部分文本内容，需要对其进行总结。总结内容应包括：主要实体信息、实体之间的关系以及规章制度的内容。请以简洁的方式进行总结，不要添加任何虚构信息。总结的长度应控制在大约1000个token左右。

以下是文本内容：
内容：{content}

现在，请对以上内容进行总结。
总结：""",

    "summarize_summary_zh":\
"""你是一位擅长总结文本内容的助手。你将收到一系列文本内容的总结，需要对这些总结进行进一步的总结。总结内容应包括：主要实体信息、实体之间的关系以及规章制度的内容。请以简洁的方式进行总结，不要添加任何虚构信息。总结的长度应控制在大约1000个token左右。

以下是总结内容：
总结：{summary}

现在，请根据问题对以上总结进行进一步的总结。
总结：""",

}