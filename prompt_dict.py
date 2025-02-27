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

Answer: """
}


"""There will be two examples:

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
"""