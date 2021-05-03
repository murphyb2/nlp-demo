# SpaCy NLP Demo

This application runs fully in the terminal and will analyze all text (.txt) files located in the ./docs folder and output the 25 most frequent words in ALL of the documents read.

This project is build using [SpaCy](https://spacy.io/).

# Setup

## Initialize Python Environment

Install the python packages as required in the Pipfile

```shell
pipenv shell
pipenv install
```

## Install SpaCy PreTrained Model

```shell
python -m spacy download en_core_web_sm
```

## Run the Script

```shell
python spNLP.py
```

# Analysis Details

After processing all the text through the SpaCy NLP pipeline, the program uses SpaCy's PhraseMatcher to find all the sentences containing the most frequent words. It matches on the word's lemma so it will find matches country -> countries.
