Prompts = {
    "summarize_details":"""You are a helpful assistant that summarizes the details of a novel.
    You will be given a part of a novel.
    You need to summarize given content.
    The summary should include the main characters, the main plot and some other details.
    You need to return the summary in a concise manner without any additional information. The length of the summary should be about 512 tokens. 
    Here is the content:
    Content: {content}
    Now, please summarize the content.
    Summary: """,

    "summarize_summary":"""You are a helpful assistant that further summarizes the summaries of a novel.
    You will be given a series of summaries of parts of a novel.
    You need to summarize the summaries in a concise manner.
    Here is the summaries:
    Summary: {summary}
    Now, please summarize the summary based on the question.
    Summary: """,

    "QA_prompt_options":"""You are a helpful assistant that answers questions about a novel.
    You will be given a novel and a question.
    You need to answer the question based on the novel.
    Here is the novel:
    Novel: {novel}
    Here is the question:
    Question: {question}
    """,

    "QA_prompt_answer":\
"""You are a helpful assistant, you are given a question, please answer the question. The answer should be a short paragraph that captures the main idea, main events, and main characters of the question. You should not include any information that is not in the question. You should only output the answer. There will be two examples:

Example - character relationship:
Question: What is the relationship between the Steve and the painter?
Evidence: 
    1. "Steve took off his coat slowly and watched his bride, Evan, as she walked towards him."
    2. "Evan open the door, a painter came in. Evan lead the painter to the room and said: 'This is my brother, he is a painter.'"

Answer: Steve is the husband of Evan, and the painter is the brother of Evan.

Example - count times:
Question: How many times did the painter meet Steve?
Evidence: 
    1. "Evan open the door, a painter came in. Evan lead the painter to the room and said: 'This is my brother, he is a painter.' Steve greeted the painter and kept doing his work."
    2. "The painter was painting the wall, and he was painting the ceiling."

Answer: The painter met Steve once.

Now, please answer the following question with the given evidences:

Question: {question}
Evidence: {evidence}

Answer: """
}