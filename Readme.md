# SpaCy NLP Demo (Web)

This is a simple Django app to render the most frequent words in a body of documents along with the documents and sentences in which they appear.

This project is build using [SpaCy](https://spacy.io/) to handle NLP and served using [Django](https://www.djangoproject.com/).

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

## Run the Server

```python
python manage.py runserver
```

# Analysis Details/Notes

After processing all the text through the SpaCy NLP pipeline, the program uses SpaCy's PhraseMatcher to find all the sentences containing the most frequent words. It matches on the word's lemma so it will find matches including country -> countries and say -> said.
