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
"""你是一位擅长回答问题的助手。你将收到一个问题和一些证据，请根据证据回答问题。答案应该是一个简短的句子，支持给定的证据，并符合问题的要求。不要假设任何证据之外的信息。你只需要输出答案。下面是两个例子：

例子 - 产品问题：

问题：申请码农贷产品的客户，至少需要提供哪些材料？
证据：2)申请[码农贷]产品的客户：
①　进入智能引导认证流程，认证完成后返回的，先按认证模式分别判断客户	是否符合准入：
a.使用百付宝公积金认证流程的，判断依据为“公积金认证准入规则”；
b.使用百付宝社保认证流程的，判断依据为“社保认证准入规则”；
c.使用个人所得税认证流程的，直接认定客户是否符合准入的结果为否；
d.使用工行账单认证流程的，直接认定客户是否符合准入的结果为否。
在判断出客户是否符合准入后，按以下逻辑处理：
是：弹出【财力证明提醒弹框】，提示客户“如果您想获得更高额度，可	以上传财力证明材料”，客户有两个选项：
直接提交：直接执行授信提交交易；
继续上传：跳转进入【授信资料上传页】。在该页面，收入证明必填，房产证明非必填，职业证明非必填。
否：判断客户预审客群是否为A类：
是：直接执行授信提交交易；
否：跳转进入【授信资料上传页】。在该页面，收入证明必填，房产证明非必填，职业证明必填。
②　进入授权查询认证流程，认证完成后返回的，先判断客户学历是否为博士/硕士/本科：
是：直接执行授信提交交易；
否：弹出【财力证明提醒弹框】，提示客户“如果您想获得更高额度，可以上传财力证明材料”，客户有两个选项：
直接提交：跳转进入【授信资料上传页】。在该页面，收入证明非必填，房产证明非必填，职业证明必填；
继续上传：跳转进入【授信资料上传页】。在该页面，收入证明必填，房产证明非必填，职业证明必填。
③　点击「跳过,继续申请」按钮，跳过认证模式的，判断客户学历是否为博士/硕士/本科：
是：直接执行授信提交交易；
否：跳转进入【授信资料上传页】，在该页面，收入证明必填，房产证明非必填，职业证明必填。

答案：1）客户选择智能信息认证流程且公积金或社保未断缴，已连续缴纳12个月，缴纳基数不低于5000的，无需提供其他材料
2）客户选择智能信息认证且预审客群为E1（即博士/硕士）的，无需提供其他材料
3）客户选择授权查询认证流程，且学历是博士/硕士/本科的，无需提供其他材料
4）客户选择授权查询认证流程，且学历不是博士/硕士/本科的，最少要提供职业证明
4）客户跳过认证模式的，且学历是博士/硕士/本科的，无需提供其他材料
5）其他情形下，客户均至少需要提供收入证明和职业证明

例子 - 产品问题:

问题：客户大专学历，在一家制造业小型工厂工作，社保公积金缴纳基数为当地最低标准，拥有国家认证的水电工高级证书，最适合他申请的产品是哪个？
证据：工人贷：
A类	：学历为硕博或一级/高级技师证书
B类：学历为本科或【二级/技师】、【三级/高级技能、高级】证书且有公积金且月缴金额≥2千（支持智能引导公积金及飞鸽公积金）
C类：学历为本科或【二级/技师】、【三级/高级技能、高级】证书	无公积金或公积金低于2千（学历为本科的非B类）
D类：学历为大专或【四级/中级技能、中级】且有公积金（支持智能引导公积金及飞鸽公积金）
E类：学历为大专或【四级/中级技能、中级】且无公积金 
或 学历为大专以下或【五级/初级技能、初级】且有公积金
F类：学历为大专以下或【五级/初级技能、初级】且无公积金

答案：工人贷

现在，请根据问题和证据回答问题：

问题：{question}
证据：{evidence}

答案：""",

    "summarize_details_zh":\
"""你是一位擅长总结文本内容的助手。你将收到一部分文本内容，需要对其进行总结。总结内容应包括：主要实体信息、实体之间的关系以及规章制度的内容。请以简洁的方式进行总结，不要添加任何虚构信息。总结的长度应控制在大约200个token左右。

以下是文本内容：
内容：{content}

现在，请对以上内容进行总结。
总结：""",

    "summarize_summary_zh":\
"""你是一位擅长总结文本内容的助手。你将收到一系列文本内容的总结，需要对这些总结进行进一步的总结。总结内容应包括：主要实体信息、实体之间的关系以及规章制度的内容。请以简洁的方式进行总结，不要添加任何虚构信息。总结的长度应控制在大约200个token左右。

以下是总结内容：
总结：{summary}

现在，请根据问题对以上总结进行进一步的总结。
总结：""",

    "rewrite_query_zh":\
"""你是一位用户查询重写助手。你将收到一条用户发送来的查询语句，这一语句可能语义并不完整或者模糊，请你根据这一语句生成一个更加清晰、准确的查询语句。

用户查询：{question}

重写后的查询：""",

    "rewrite_query_zh_v2":\
"""你是一位用户查询重写助手。你将收到一条用户发送来的查询语句，这一语句可能语义并不完整或者模糊，请你根据这一语句生成一个更加清晰、准确的查询语句。请仅输出你改写后的查询语句，不要带任何额外的解释。如果用户语义非常模糊，则不能捏造任何用户没有提到的信息，单纯返回其原本的查询内容。

用户查询：{question}""",

    "extract_entities_json":\
"""你是一名信息抽取助手，请从给定文本中找出关键实体（人物、机构、地点、产品、事件核心对象等）。只输出 JSON，格式如下：
{{
  "entities": [
    {{"name": "实体名称", "type": "可选的类型说明"}}
  ]
}}

要求：
- 仅保留文本中真实出现过的实体，不要编造。
- name 字段保持原文，不要拆分或翻译。
- 不要输出除 JSON 以外的任何内容、标点或解释。

文本：{content}
JSON：""",

    "verify_relation_json":\
"""你是一名关系判定助手。请根据全文判断两个实体是否存在直接关联（不限于同一句）。只输出 JSON，格式如下：
{{
  "related": true/false,
  "reason": "可选，简要说明依据的原文线索"
}}

要求：
- related 仅为 true/false，避免其他值。
- 若文中无任何关联证据，则输出 false。
- 不要输出除 JSON 以外的任何内容。

全文：{content}
实体1：{entity1}
实体2：{entity2}
JSON：""",

    "extract_graph_relations_json":\
"""你是一名信息抽取助手，请从全文中抽取实体与实体间的关系。仅输出 JSON，格式如下：
{{
  "entities": [
    {{"name": "实体名称", "type": "可选类型"}}
  ],
  "relations": [
    {{"head": "实体1", "tail": "实体2", "type": "可选关系类型", "evidence": "可选原文片段"}}
  ]
}}

要求：
- 仅保留原文出现的实体与关系，不要编造。
- name/head/tail 使用原文，不要翻译或拆分。
- 如果没有检测到关系，relations 输出空数组。
- 不要输出除 JSON 以外的内容。

全文：{content}
JSON：""",
}