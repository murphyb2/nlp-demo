from typing import Dict
import spacy
from collections import Counter
from spacy.matcher import PhraseMatcher
from spacy.tokens import DocBin
import pandas as pd
import os

def loadDocs(nlp) -> DocBin:

    directory = './docs'
    docBin = DocBin()
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and filename.endswith('.txt'):
            print(f"Reading {filename.split('.')[0]}")
            # Read doc
            with open(f, encoding="utf8") as d:
                text = d.read()
            newDoc = nlp(text)
            newDoc.user_data = {"name": filename.split('.')[0]}

            docBin.add(newDoc)
    return docBin

def mostFrequentWords(docBin: DocBin, nlp) -> Counter():
    word_freq = Counter()
    for doc in list(docBin.get_docs(nlp.vocab)):
        # Find the most common words in the doc that aren't stop words and are alphabetical
        # Take the lemma of the word
        words = [token.lemma_ for token in doc if token.is_stop is False and token.is_alpha and not token.is_punct]
        
        word_freq.update(words)
    return word_freq

def matchToDict()-> Dict:
    pass

def main() -> None:
    
    print(">>>>>>>>                                                                         <<<<<<<<")
    print(">>>>>>>>                      SpaCy Word Frequency Demo                          <<<<<<<<")
    print(">>>>>>>>                                                                         <<<<<<<<")
    print()
    print("> This application will analyze all text (.txt) files located in the docs folder")
    print("> and output the 25 most frequent words in ALL of the documents.")
    print()

    result = input("Press any key to begin analysis or 'z' to quit...\n")
    if result.lower() == 'z':
        return
    print()

    # Load the spacy model and read the doc text
    nlp = spacy.load("en_core_web_sm")
    
    docBin = loadDocs(nlp)

    print()
    print("Analyzing...")
    print()
    
    # Count most frequent words
    word_freq = mostFrequentWords(docBin, nlp)
        
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

    print()
    answer = input("Save to csv file? Press 'Y' or 'N':\n")
    if answer.lower() == 'y':
        print("Saving to out.csv...")
        df.to_csv("out.csv")
    
if __name__ == "__main__":
    main()