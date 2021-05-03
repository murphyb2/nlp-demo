import spacy
from collections import Counter
from spacy.matcher import PhraseMatcher
from spacy.tokens import DocBin
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

    print()
    print("Analyzing...")
    print()
    
    # Count most frequent words
    word_freq = Counter()
    for doc in list(docBin.get_docs(nlp.vocab)):
        # Find the most common words in the doc that aren't stop words and are alphabetical
        # Take the lemma of the word
        words = [token.lemma_ for token in doc if token.is_stop is False and token.is_alpha and not token.is_punct]
        
        word_freq.update(words)
        
    # Use matcher to find sentences containing most common words
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    pattern = [nlp(t[0]) for t in word_freq.most_common(25)]

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
            w = matched_span.lemma_
            
            if w in groups:
                groups[w]["sents"].add(matched_span.sent.text)
                groups[w]["docs"].add(d.user_data["name"])
            else:
                groups[w] = {
                    "word": w,
                    "sents": { matched_span.sent.text },
                    "docs": { d.user_data["name"] },
                    "freq": word_freq[w]
                }

    listWords = list(groups.values())
    sortedList = sorted(listWords, key=lambda x: x["freq"], reverse=True)
        
    df = pd.DataFrame(sortedList)
    df.sort_values(by=["freq"], inplace=True, ascending=False)
    print(df)
    df.to_csv("out.csv")
    
if __name__ == "__main__":
    main()