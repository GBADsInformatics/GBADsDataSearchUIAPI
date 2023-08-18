import spacy
import csv
import nltk
import numpy as np
import sys
import os
import datetime

# Add the project root directory to the Python path
root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_directory)

# Now you can import the module directly
import ProcessSearch as PS

nltk.download("stopwords")

# Load the spaCy English language model
nlp = spacy.load("en_core_web_lg")

# Category -> words
data = {
    "Names": ["john", "jay", "dan", "nathan", "bob"],
    "Colors": ["yellow", "red", "green"],
    "Places": ["tokyo", "beijing", "washington", "mumbai", "ethiopia", "canada", "sub-saharan africa"],
    "Species": ["cows", "chickens", "poultry", "bovine", "horses"],
    "Years": ["2001", "1971", "96", "2000s", "93'"],
    "General": ["the", "by", "here", "population", "random", "tile", "canda"],
    "Regions": ["central asia", "latin america", "oceania", "caribbean"],
    "Mistakes": ["rusia", "subsaharan", "saharan"],
}

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

# Good test to implement:
# Bovine populaton in Canda and Rusia in 2009 or 2011
test_queries = [
    "Chickens in Great Britain between 2011 and 2010",
    "Poultry in Canada in 2019",
    "Canadian Poultry population",
    "Ethiopia hens in 1997",
    "Chinese bovine in 96",
    "The population of cows in vietnam in the 2000s",
    "Horses in 75",
    "Random search that is nonsense",
    "Poultry and Bovine population in Canada and Russia in 2011 and 2010",
    "Bovine populaton in Canda and Rusia in 2009 or 2011",
    "camel population in Subsaharan Africa",
    "latest data on goats in Canada",
    "cattle in Africa this year",
    ""
]


def test_ner():
    ner = PS.NER(
        nlp, data, categories, embeddings_index, data_embeddings, nationality_mapping
    )
    # Expected results for each test query

    current_year = datetime.datetime.now().year
    current_year_str = str(current_year)

    expected_results = [
        {"species": ["Chickens"], "year": ["2011", "2010"], "country": ["Great britain"]},
        {"species": ["Poultry"], "year": ["2019"], "country": ["Canada"]},
        {"species": ["Poultry"], "year": [], "country": ["Canada"]},
        {"species": ["Hens"], "year": ["1997"], "country": ["Ethiopia"]},
        {"species": ["Bovine"], "year": ["96"], "country": ["China"]},
        {"species": ["Cows"], "year": ["2000s"], "country": ["Vietnam"]},
        {"species": ["Horses"], "year": ["75"], "country": []},
        {"species": [], "year": [], "country": []},
        # Multiple speciies, years, and countries
        {"species": ["Poultry", "Bovine"], "year": ["2011", "2010"], "country": ["Canada", "Russia"]},
        {"species": ["Bovine"], "year": ["2009", "2011"], "country": []},
        {"species": ["Camel"], "year": [], "country": ["Subsaharan africa"]},
        {"species": ["Goats"], "year": [current_year_str], "country": ["Canada"]},
        {"species": ["Cattle"], "year": [current_year_str], "country": ["Africa"]},
        {"species": [], "year": [], "country": []}
    ]

    for i in range(0, len(test_queries)):
        result = ner.perform_ner(test_queries[i])
        print(result)
        assert result == expected_results[i]

    # for query, expected_result in zip(test_queries, expected_results):
    #     result = ner.perform_ner(query)
    #     print(result)
    #     assert result == expected_result
