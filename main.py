import spacy
import csv
import nltk
import numpy as np
import ProcessSearch as PS
from fastapi import FastAPI

nltk.download("stopwords")

# Load the spaCy English language model
nlp = spacy.load("en_core_web_lg")

# Category -> words
data = {
    "Names": ["john", "jay", "dan", "nathan", "bob"],
    "Colors": ["yellow", "red", "green"],
    "Places": ["tokyo", "beijing", "washington", "mumbai"],
    "Species": ["cows", "chickens", "poultry", "bovine", "horses"],
    "Years": ["2001", "1971", "96", "2000s", "93'"],
}

# test_queries = [
#     "Chickens in Great Britain between 2011 and 2010",
#     "Poultry in Canada in 2019",
#     "Canadian Poultry population",
#     "Ethiopia hens in 1997",
#     "Chinese bovine in 96",
#     "The population of cows in vietnam in the 2000s",
#     "Horses in 75",
#     "Random search that is nonsense"
# ]

# Words -> category
categories = {word: key for key, words in data.items() for word in words}

# Load the whole embedding matrix
embeddings_index = {}
with open("glove.6B.50d.txt") as f:
    for line in f:
        values = line.split()
        word = values[0]
        embed = np.array(values[1:], dtype=np.float32)
        embeddings_index[word] = embed
print("Loaded %s word vectors." % len(embeddings_index))

# Embeddings for available words
data_embeddings = {
    key: value for key, value in embeddings_index.items() if key in categories.keys()
}

# Load nationality mapping from CSV file
nationality_mapping = {}
with open("nationality.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        nationality_mapping[row["nationality"].lower()] = row["en_short_name"]


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(query: str):
    ner = PS.NER(
        nlp, data, categories, embeddings_index, data_embeddings, nationality_mapping
    )
    result = ner.perform_ner(query)
    return result
