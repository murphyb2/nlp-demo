import spacy
from collections import Counter
from spacy.matcher import Matcher
from spacy.tokens import DocBin
from spacy.attrs import LEMMA, ORTH
import pandas as pd

def main() -> None:
    docBin = DocBin()
    # Load the spacy model and read the doc text
    nlp = spacy.load("en_core_web_sm")

    for i in range(1,7):
        print(f"Reading doc {i}...")
        # Read doc
        with open(f'./docs/doc{i}.txt', encoding="utf8") as d:
            text = d.read()
        newDoc = nlp(text)
        newDoc.user_data = {"name": f"doc{i}"}

        docBin.add(newDoc)

    print("Analyzing...")
    word_freq = Counter()
    for doc in list(docBin.get_docs(nlp.vocab)):
        print(doc.user_data["name"])
        # Find the most common words in the doc that aren't stop words and are alphabetical
        # Take the lemma of the word
        words = [token.lemma_ for token in doc if token.is_stop is False and token.is_alpha and not token.is_punct]
        
        word_freq.update(words)
    
    for t in word_freq.most_common(25):
        print(f">> {t[0]}, {t[1]}")
        
    # # Use matcher to find sentences containing most common words
    matcher = Matcher(nlp.vocab)
    pattern = [[{"ORTH": t[0],} ] for t in word_freq.most_common(25)]

    matcher.add('MOST_FREQ', pattern)

    # Construct dict with most frequent words and 
    # the sentences they appear in, 
    # the docs they appear in, 
    # and the frequency of the word
    groups = {}
    for d in list(docBin.get_docs(nlp.vocab)):
        matches = matcher(d)
        for match_id, start, end in matches:
            # Get matched span
            matched_span = d[start:end]
            if matched_span.text in groups:
                groups[matched_span.text]["sents"].append(matched_span.sent)
                groups[matched_span.text]["docs"].add(d.user_data["name"])
            else:
                groups[matched_span.text] = {
                    "sents": [matched_span.sent],
                    "docs": { d.user_data["name"] },
                    "freq": word_freq[matched_span.text]
                }

    for word, data in groups.items():
        print(word, data["freq"])
        
    
    # df = pd.DataFrame(groups)
    # print(df.head)
    
if __name__ == "__main__":
    main()