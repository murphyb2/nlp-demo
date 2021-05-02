import spacy
from collections import Counter
from spacy.matcher import Matcher
from spacy.tokens import DocBin
import pandas as pd

def main() -> None:
    docBin = DocBin()
    # Load the spacy model and read the doc text
    nlp = spacy.load("en_core_web_sm")

    for i in range(1,7):
        print(f"doc {i}")
        # Read doc
        with open(f'./docs/doc{i}.txt', encoding="utf8") as d:
            text = d.read()

        docBin.add(nlp(text))

    word_freq = Counter()
    for doc in list(docBin.get_docs(nlp.vocab)):
    
        # Find the most common words in the doc that aren't stop words and are alphabetical
        words = [token.lemma_ for token in doc if token.is_stop is False and token.is_alpha]
        word_freq.update(words)
        # print(word_freq.most_common(10))
    
    for t in word_freq.most_common(25):
        print(f">> {t[0]}, {t[1]}")
        
    # # Use matcher to find sentences containing most common words
    # matcher = Matcher(nlp.vocab)
    # pattern = [[{"LOWER": t[0]}] for t in word_freq.most_common(5)]
    # print(pattern)
    # matcher.add('MOST_FREQ', pattern)

    # matches = matcher(doc)

    # groups = {}
    # for match_id, start, end in matches:
    #     # Get matched span
    #     matched_span = doc[start:end]
    #     if matched_span.text in groups:
    #         groups[matched_span.text].append(matched_span.sent)
    #     else:
    #         groups[matched_span.text] = [matched_span.sent]

    #     # print(matched_span.text)
    #     # print(matched_span.sent)
    #     # print("-------------------------------------")
    
    # for word, sents in groups.items():
    #     print(word)
    #     print(len(sents))
    
    # df = pd.DataFrame(groups)
    # print(df.head)
    
if __name__ == "__main__":
    main()