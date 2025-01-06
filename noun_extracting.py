from yb_dataloader import NarrativeQALoader, NovelQALoader

import nltk

def extract_nouns(text):
    # download the necessary nltk data.
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('averaged_perceptron_tagger')
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

    # split the text into sentences.
    sentences = nltk.sent_tokenize(text)
    
    # store the noun and its cooccurrence information.
    noun_pairs = {}
    all_nouns = set()
    
    for sentence in sentences:
        # tokenize the sentence and tag the words.
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        
        # extract the nouns in the sentence.
        sentence_nouns = [word.lower() for word, tag in tagged if tag.startswith('NN')]
        all_nouns.update(sentence_nouns)
        
        # count the cooccurrence of nouns.
        for i in range(len(sentence_nouns)):
            for j in range(i+1, len(sentence_nouns)):
                noun1, noun2 = sorted([sentence_nouns[i], sentence_nouns[j]])
                pair = (noun1, noun2)
                noun_pairs[pair] = noun_pairs.get(pair, 0) + 1
    
    return list(all_nouns), noun_pairs

def get_noun_cooccurrence(text):
    """get the cooccurrence of nouns in the text."""
    nouns, cooccurrence = extract_nouns(text)
    
    # sort the cooccurrence by the frequency.
    sorted_pairs = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "nouns": nouns,
        "cooccurrence": sorted_pairs
    }

if __name__ == "__main__":
    text = "The cat sat on the mat. The dog chased the cat. The cat ran away."
    print(get_noun_cooccurrence(text))