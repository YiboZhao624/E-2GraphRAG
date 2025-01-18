BIO_TRIPLETS_PROMPT = """You are a helpful assistant, you are given a text, please extract the triplets from the text. There will be two examples:

Example 1:
Text: Various methods of diagnosing allergic factors in chronic rhinitis are discussed. Among the procedures which aim at detecting specific allergens, i.e. skin testing, RAST, and nasal provocation tests, the last mentioned, as they are performed directly on the shock organ, have so far been found to give the most accurate picture of clinically dominant allergens and of the intensity and character of the rhinitis. However, information obtained by analysing the correlations between different procedures is not unanimous. As long as test techniques and allergen extracts have not been standardized, one particular test cannot be recommended as the method of choice.

Triplets:(skin testing, aims at detecting, specific allergens)  
(RAST, aims at detecting, specific allergens)  
(nasal provocation tests, aims at detecting, specific allergens)  
(nasal provocation tests, performed directly on, shock organ)  
(nasal provocation tests, give, accurate picture of clinically dominant allergens)  
(nasal provocation tests, give, accurate picture of intensity and character of rhinitis)  
(correlations, analysed between, different procedures)  
(test techniques, lack, standardization)  
(allergen extracts, lack, standardization)  
(test techniques, not recommended as, method of choice)  


Example 2:
Text: KRAS genes belong to the most frequently mutated family of oncogenes in cancer. The G12C mutation, found in a third of lung, half of colorectal and pancreatic cancer cases, is believed to be responsible for a substantial number of cancer deaths. For 30 years, KRAS has been the subject of extensive drug-targeting efforts aimed at targeting KRAS protein itself, but also its post-translational modifications, membrane localization, protein-protein interactions and downstream signalling pathways. So far, most KRAS targeting strategies have failed, and there are no KRAS-specific drugs available. However, clinical candidates targeting the KRAS G12C protein have recently been developed. MRTX849 and recently approved Sotorasib are covalent binders targeting the mutated cysteine 12, occupying Switch II pocket.Herein, we describe two fragment screening drug discovery campaigns that led to the identification of binding pockets on the KRAS G12C surface that have not previously been described. One screen focused on non-covalent binders to KRAS G12C, the other on covalent binders.

Triplets:
(KRAS genes, belong to, oncogenes)  
(KRAS genes, frequently mutated in, cancer)  
(G12C mutation, found in, lung cancer)  
(G12C mutation, found in, colorectal cancer)  
(G12C mutation, found in, pancreatic cancer)  
(G12C mutation, responsible for, cancer deaths)  
(KRAS, subject of, drug-targeting efforts)  
(drug-targeting efforts, aimed at targeting, KRAS protein)  
(drug-targeting efforts, targeted, post-translational modifications)  
(drug-targeting efforts, targeted, membrane localization)  
(drug-targeting efforts, targeted, protein-protein interactions)  
(drug-targeting efforts, targeted, downstream signalling pathways)  
(KRAS targeting strategies, failed, most cases)  
(KRAS-specific drugs, unavailable for, KRAS)  
(clinical candidates, developed for, KRAS G12C protein)  
(MRTX849, targets, mutated cysteine 12)  
(Sotorasib, targets, mutated cysteine 12)  
(Sotorasib, approved for, KRAS G12C protein)  
(fragment screening campaigns, identified, new binding pockets)  
(binding pockets, located on, KRAS G12C surface)  
(one screen, focused on, non-covalent binders)  
(other screen, focused on, covalent binders)  

Now, please extract the triplets from the following text:

Text: {text}

Triplets:
"""


NOVEL_TRIPLETS_PROMPT = \
"""You are a professional information extraction assistant. Please extract relationship triplets from the given text. Do not output any other information.

Triplet Definition:
- Triplets can be used to describe the relationship between two entities, or the relationship between an entity and a property. Do NOT use the pronoun like "I", "you", "he", "she", "it", "they", "we" etc in the triplets. When the context include the pronoun, please find the noun phrase that the pronoun refers to and replace the pronoun with the noun phrase.
- Head: Important noun phrases from the text
- Relation: Concise verb or prepositional phrases describing the relationship
- Tail: Important noun phrases from the text

Format Requirements:
- Each triplet should be written as (head, relation, tail)
- One triplet per line
- Entities should be key terms or phrases from the text
- Relations should accurately reflect the semantic connection

Example 1 - Someones speech:
Text: "The new smartphone features a high-resolution display. Its battery life lasts for two days, and users praise its camera quality. I have bought such a phone yesterdays." Tomas said.

Triplets:
(smartphone, features, high-resolution display)
(battery life, lasts for, two days)
(users, praise, camera quality)
(Tomas, bought, smartphone)

Example 2 - Scientific Finding:
Text: Recent studies show that regular exercise improves cognitive function. Additionally, physical activity reduces the risk of cardiovascular disease and helps maintain healthy body weight.

Triplets:
(regular exercise, improves, cognitive function)
(physical activity, reduces risk of, cardiovascular disease)
(physical activity, helps maintain, healthy body weight)

Example 3 - Event:
Text: Wendy met a boy in the park, and they went to the beach together. They had a great time. By the end of the day, Wendy asked for the boy's phone number, and finally found out that he was her ex-boyfriend, Tom, who she had not seen for years.

Triplets:
(Wendy, met, Tom)
(Wendy, went to the beach with, Tom)
(Wendy, asked for, Tom's phone number)
(Wendy, is ex-girlfriend of, Tom)

Now, please extract relationship triplets from the following text:

Text: {text}

Triplets:"""

# not used.
NOVEL_TRIPLETS_PROMPT_WO_RELATION = """You are a professional information extraction assistant. Please extract relationship triplets from the given text. Do not output any other information.

Triplet Definition:
- Triplets can be used to describe the relationship between two entities, or the relationship between an entity and a property. Do NOT use the pronoun like "I", "you", "he", "she", "it", "they", "we" etc in the triplets. When the context include the pronoun, please find the noun phrase that the pronoun refers to and replace the pronoun with the noun phrase.
- Head: Important noun phrases from the text
- Relation: Concise verb or prepositional phrases describing the relationship
- Tail: Important noun phrases from the text

Format Requirements:
- Each triplet should be written as (head, relation, tail)
- One triplet per line
- Entities should be key terms or phrases from the text
- Relations should accurately reflect the semantic connection

Example 1 - Someones speech:
Text: "The new smartphone features a high-resolution display. Its battery life lasts for two days, and users praise its camera quality. I have bought such a phone yesterdays." Tomas said.

Triplets:
(smartphone, features, high-resolution display)
(battery life, lasts for, two days)
(users, praise, camera quality)
(Tomas, bought, smartphone)

Example 2 - Scientific Finding:
Text: Recent studies show that regular exercise improves cognitive function. Additionally, physical activity reduces the risk of cardiovascular disease and helps maintain healthy body weight.

Triplets:
(regular exercise, improves, cognitive function)
(physical activity, reduces risk of, cardiovascular disease)
(physical activity, helps maintain, healthy body weight)

Example 3 - Event:
Text: Wendy met a boy in the park, and they went to the beach together. They had a great time. By the end of the day, Wendy asked for the boy's phone number, and finally found out that he was her ex-boyfriend, Tom, who she had not seen for years.

Triplets:
(Wendy, met, Tom)
(Wendy, went to the beach with, Tom)
(Wendy, asked for, Tom's phone number)
(Wendy, is ex-girlfriend of, Tom)

Now, please extract relationship triplets from the following text:

Text: {text}

Triplets:"""

SUMMARY_PROMPT = """You are a helpful assistant, you are given a text, please summarize the text. The summary should be a short paragraph that captures the main idea, main events, and main characters of the text. You should not include any information that is not in the text. You should only output the summary. There will be an example:

Example - Pride and Prejudice:
Text: It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.
However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.
'My dear Mr. Bennet,' said his lady to him one day, 'have you heard that Netherfield Park is let at last?'
Mr. Bennet replied that he had not.
'But it is,' returned she; 'for Mrs. Long has just been here, and she told me all about it.'
Mr. Bennet made no answer.
'Do you not want to know who has taken it?' cried his wife impatiently.
'You want to tell me, and I have no objection to hearing it.'
This was invitation enough.
'Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it, that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week.'
'What is his name?'
'Bingley.'
'Is he married or single?'
'Oh! Single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for our girls!'
'How so? How can it affect them?'
'My dear Mr. Bennet,' replied his wife, 'how can you be so tiresome! You must know that I am thinking of his marrying one of them.'
'Is that his design in settling here?'
'Design! Nonsense, how can you talk so! But it is very likely that he may fall in love with one of them, and therefore you must visit him as soon as he comes.'
'I see no occasion for that. You and the girls may go, or you may send them by themselves, which perhaps will be still better, for as you are as handsome as any of them, Mr. Bingley might like you the best of the party.'
'My dear, you flatter me. I certainly have had my share of beauty, but I do not pretend to be anything extraordinary now. When a woman has five grown-up daughters, she ought to give over thinking of her own beauty.'
'In such cases, a woman has not often much beauty to think of.'
'But, my dear, you must indeed go and see Mr. Bingley when he comes into the neighbourhood.'
'It is more than I engage for, I assure you.'
'But consider your daughters. Only think what an establishment it would be for one of them. Sir William and Lady Lucas are determined to go, merely on that account, for in general, you know, they visit no new comers. Indeed you must go, for it will be impossible for us to visit him if you do not.'
'You are over-scrupulous, surely. I dare say Mr. Bingley will be very glad to see you; and I will send a few lines by you to assure him of my hearty consent to his marrying whichever he chooses of the girls; though I must throw in a good word for my little Lizzy.'
'I desire you will do no such thing. Lizzy is not a bit better than the others; and I am sure she is not half so handsome as Jane, nor half so good-humoured as Lydia. But you are always giving her the preference.'
'They have none of them much to recommend them,' replied he; 'they are all silly and ignorant like other girls; but Lizzy has something more of quickness than her sisters.'
'Mr. Bennet, how can you abuse your own children in such a way? You take delight in vexing me. You have no compassion for my poor nerves.'
'You mistake me, my dear. I have a high respect for your nerves. They are my old friends. I have heard you mention them with consideration these last twenty years at least.'
'Ah, you do not know what I suffer.'
'But I hope you will get over it, and live to see many young men of four thousand a year come into the neighbourhood.'
'It will be no use to us, if twenty such should come, since you will not visit them.'
'Depend upon it, my dear, that when there are twenty, I will visit them all.'
Mr. Bennet was so odd a mixture of quick parts, sarcastic humour, reserve, and caprice, that the experience of three-and-twenty years had been insufficient to make his wife understand his character. Her mind was less difficult to develop. She was a woman of mean understanding, little information, and uncertain temper. When she was discontented, she fancied herself nervous. The business of her life was to get her daughters married; its solace was visiting and news.

Summary:
This passage revolves around a conversation between Mr. and Mrs. Bennet regarding the arrival of a wealthy young man, Mr. Bingley, in their neighborhood. Mrs. Bennet is excited by the prospect of Mr. Bingley marrying one of their five daughters, seeing it as an opportunity to secure a prosperous future for the family. She eagerly shares the news with her husband, emphasizing Mr. Bingley’s wealth and single status. Mr. Bennet, however, responds with sarcasm and indifference, teasing his wife about her obsession with marrying off their daughters. Their exchange highlights the contrasting personalities of the couple: Mrs. Bennet is portrayed as overly concerned with social status and marriage, while Mr. Bennet is detached and amused by her preoccupations. The dialogue also subtly introduces the family dynamics and the societal expectations of the time, particularly the pressure on women to find suitable husbands.

New Text:
{text}

Summary:
"""


NER_PROMPT = """You are a helpful assistant, you are given a question, please extract the entities from the question. And the extracted entities should be in the format of [entity1, entity2, ...]. There will be three examples:

Example 1:
Question: Who is Madame Mirliflore?

Entities:
[Madame Mirliflore]

Example 2:
Question: What happened between the Tom and the painter?

Entities:
[Tom, painter]

Example 3:
Question: What is the dress color of Jimmy when he was working in the factory?

Entities:
[Jimmy, factory]

Now, please extract the entities from the following question:

Question: {question}

Entities:
"""


QA_PROMPT_RAPTOR = """You are a helpful assistant, you are given a question, please answer the question. The answer should be a short paragraph that captures the main idea, main events, and main characters of the question. You should not include any information that is not in the question. You should only output the answer. There will be an example:

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

Now, please answer the following question:

Question: {question}
Evidence: {evidence}

Answer:
"""