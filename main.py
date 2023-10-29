import spacy
import csv
import nltk
import numpy as np
import ProcessSearch as PS
from fastapi import FastAPI, APIRouter
import os
import Autocomplete as AC
from fastapi.middleware.cors import CORSMiddleware

nltk.download("stopwords")
nltk.download("punkt")

# Load the spaCy English language model
nlp = spacy.load("en_core_web_lg")

# Category -> words
data = {
    "Names": ["john", "jay", "dan", "nathan", "bob"],
    "Continents": ["asia", "north america", "south america", "europe", "oceania", "antarctica", "africa"],
    "Places": ["tokyo", "beijing", "washington", "mumbai", "ethiopia", "canada", "sub-saharan africa", "madagascar"],
    "Species": ["cows", "chickens", "poultry", "bovine", "horses", "tigers", "puffins", "koalas", "lion", "hawks"],
    "Years": ["2001", "1971", "96", "2000s", "93'"],
    "General": ["the", "by", "here", "population", "random", "tile", "canda"],
    "Regions": ["central asia", "latin america", "oceania", "caribbean"],
    "Mistakes": ["rusia", "subsaharan", "saharan"],
}

# Words -> category
categories = {word: key for key, words in data.items() for word in words}

# Load the whole embedding matrix
embeddings_index = {}

# For autocomplete module
words = []

with open("glove.6B.50d.txt") as f:
    for line in f:
        values = line.split()
        word = values[0]
        words.append(word)
        embed = np.array(values[1:], dtype=np.float32)
        embeddings_index[word] = embed


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


# For autocomplete object
# This is our vocabulary
V = set(words)

BASE_URL = os.environ.get("BASE_URL", "")
app = FastAPI(docs_url=BASE_URL + "/docs", openapi_url=BASE_URL + "/openapi.json")
router = APIRouter(prefix=BASE_URL)

# Add CORS for webapp compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@router.get("/", tags=['Ping'])
def test_api_connection():
    return "pong"


@router.get("/search", tags=['Search'])
def perform_a_search_query(query: str):
    ner = PS.NER(
        nlp, data, categories, embeddings_index, data_embeddings, nationality_mapping
    )
    autocorrect = AC.Autocomplete(V)
    result = ner.perform_ner(query)
    ac_return = autocorrect.check_sentence(query)
    print(ac_return)
    return result


# This router allows a custom path to be used for the API
app.include_router(router)
