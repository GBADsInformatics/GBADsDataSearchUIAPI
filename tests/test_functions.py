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
with open("glove.6B.200d.txt") as f:
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
    "cattle in South Africa this year",
    "",
    "Sheep and cattle in New Zealand in the 90s",
    "Poultry and sheep in Australia in the 1980s",
    "Cattle in Brazil between 1998 and 2002",
    "Duck population in China in 2015",
    "Swine in Germany in 2000",
    "Elephants in Kenya in the 2010s",
    "Buffalo in India in 85",
    "Lion population in Africa",
    "Birds and rabbits in the United States in 2018",
    "Cat populaton in Italy and Spain in 2005 or 2006",
    "Giraffe population in East Africa",
    "latest data on horses in Mexico",
    "dogs in Asia this year",

    "Cats in Japan in 2015 and 2016",
    "Dogs in France in 2020",
    "Rabbits in the United Kingdom",
    "Sheep population in New Zealand in the 80s",
    "Goats in India between 2005 and 2010",
    "Birds in the United States in the 21st century",
    "Pigs in Germany in 1999",
    "Tigers in India in 1990",
    "Elephants in Thailand in the 2000s",
    "Horses in Argentina in 2017",
    "Lion population in South Africa",
    "Ducks in Canada in 2014",
    "Poultry in Russia in 2012",
    "Camels in Saudi Arabia in 1995",
    "Giraffes in Kenya in the 1990s",
    "Buffalo in Nepal in 1988",
    "Kangaroos in Australia",
    "Monkeys in Brazil in 2007",
    "Whales in Antarctica in 2021",
    "Sharks in the Pacific Ocean",

    "Pandas in China in 2010",
    "Koalas in Australia in 2022",
    "Penguins in Antarctica in 1995",
    "Kangaroos in Australia in the 2000s",
    "Turtles in Mexico in 2019",
    "Bears in Canada in the 1980s",
    "Lemurs in Madagascar in 2015, 2016, and 2017",
    "Seals in The Arctic",
    "Cheetahs in Namibia in 1998",
    "Dolphins in The Bahamas in 2005",
    "Puffins in Iceland",
    "Otters in Canada in 2023",
    # "Polar bears in The Arctic in the 1980s",
    "Koalas in Australia in the 1990s",
    "Giraffes in Kenya in 2020",
    "Leopards in Africa",
    "Pandas",
    "2000",
    # "The rainforest",
    "Hawks in the United States in 2021",
    # "Polar bears in The Arctic in the 2010s",
    "Penguins in Antarctica in the 1990s",
    # "Sharks in The Pacific Ocean in 2012",
    "Crocodiles in Australia in 2001",
    "Seals in Greenland",
    "Kangaroos",
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
        {"species": ["Cattle"], "year": [current_year_str], "country": ["South africa"]},
        {"species": [], "year": [], "country": []},
        {"species": ["Sheep", "Cattle"], "year": ["90s"], "country": ["New zealand"]},
        {"species": ["Poultry", "Sheep"], "year": ["1980s"], "country": ["Australia"]},
        {"species": ["Cattle"], "year": ["1998", "2002"], "country": ["Brazil"]},
        {"species": ["Duck"], "year": ["2015"], "country": ["China"]},
        {"species": ["Swine"], "year": ["2000"], "country": ["Germany"]},
        {"species": ["Elephants"], "year": ["2010s"], "country": ["Kenya"]},
        {"species": ["Buffalo"], "year": ["85"], "country": ["India"]},
        {"species": ["Lion"], "year": [], "country": ["Africa"]},
        {"species": ["Birds", "Rabbits"], "year": ["2018"], "country": ["The united states"]},
        {"species": ["Cat"], "year": ["2005", "2006"], "country": ["Italy", "Spain"]},
        {"species": ["Giraffe"], "year": [], "country": ["East africa"]},
        {"species": ["Horses"], "year": [current_year_str], "country": ["Mexico"]},
        {"species": ["Dogs"], "year": [current_year_str], "country": ["Asia"]},

        {"species": ["Cats"], "year": ["2015", "2016"], "country": ["Japan"]},
        {"species": ["Dogs"], "year": ["2020"], "country": ["France"]},
        {"species": ["Rabbits"], "year": [], "country": ["The united kingdom"]},
        {"species": ["Sheep"], "year": ["80s"], "country": ["New zealand"]},
        {"species": ["Goats"], "year": ["2005", "2010"], "country": ["India"]},
        {"species": ["Birds"], "year": [], "country": ["The united states"]},
        {"species": ["Pigs"], "year": ["1999"], "country": ["Germany"]},
        {"species": ["Tigers"], "year": ["1990"], "country": ["India"]},
        {"species": ["Elephants"], "year": ["2000s"], "country": ["Thailand"]},
        {"species": ["Horses"], "year": ["2017"], "country": ["Argentina"]},
        {"species": ["Lion"], "year": [], "country": ["South africa"]},
        {"species": ["Ducks"], "year": ["2014"], "country": ["Canada"]},
        {"species": ["Poultry"], "year": ["2012"], "country": ["Russia"]},
        {"species": ["Camels"], "year": ["1995"], "country": ["Saudi arabia"]},
        {"species": ["Giraffes"], "year": ["1990s"], "country": ["Kenya"]},
        {"species": ["Buffalo"], "year": ["1988"], "country": ["Nepal"]},
        {"species": ["Kangaroos"], "year": [], "country": ["Australia"]},
        {"species": ["Monkeys"], "year": ["2007"], "country": ["Brazil"]},
        {"species": ["Whales"], "year": ["2021"], "country": ["Antarctica"]},
        {"species": ["Sharks"], "year": [], "country": ["The pacific ocean"]},

        {"species": ["Pandas"], "year": ["2010"], "country": ["China"]},
        {"species": ["Koalas"], "year": ["2022"], "country": ["Australia"]},
        {"species": ["Penguins"], "year": ["1995"], "country": ["Antarctica"]},
        {"species": ["Kangaroos"], "year": ["2000s"], "country": ["Australia"]},
        {"species": ["Turtles"], "year": ["2019"], "country": ["Mexico"]},
        {"species": ["Bears"], "year": ["1980s"], "country": ["Canada"]},
        {"species": ["Lemurs"], "year": ["2015", "2016", "2017"], "country": ["Madagascar"]},
        {"species": ["Seals"], "year": [], "country": ["Arctic"]},
        {"species": ["Cheetahs"], "year": ["1998"], "country": ["Namibia"]},
        {"species": ["Dolphins"], "year": ["2005"], "country": ["Bahamas"]},
        {"species": ["Puffins"], "year": [], "country": ["Iceland"]},
        {"species": ["Otters"], "year": ["2023"], "country": ["Canada"]},
        # {"species": ["Polar bears"], "year": ["1980s"], "country": ["Arctic"]},
        {"species": ["Koalas"], "year": ["1990s"], "country": ["Australia"]},
        {"species": ["Giraffes"], "year": ["2020"], "country": ["Kenya"]},
        {"species": ["Leopards"], "year": [], "country": ["Africa"]},
        {"species": ["Pandas"], "year": [], "country": []},
        {"species": [], "year": ["2000"], "country": []},
        # {"species": [], "year": [], "country": ["The rainforest"]},
        {"species": ["Hawks"], "year": ["2021"], "country": ["The united states"]},
        # {"species": ["Polar bears"], "year": ["2010s"], "country": ["Arctic"]},
        {"species": ["Penguins"], "year": ["1990s"], "country": ["Antarctica"]},
        # {"species": ["Sharks"], "year": ["2012"], "country": ["The Pacific Ocean"]},
        {"species": ["Crocodiles"], "year": ["2001"], "country": ["Australia"]},
        {"species": ["Seals"], "year": [], "country": ["Greenland"]},
        {"species": ["Kangaroos"], "year": [], "country": []},
    ]

    for i in range(0, len(test_queries)):
        result = ner.perform_ner(test_queries[i])
        print(result)
        print(expected_results[i])
        print(result==expected_results[i])
        assert result == expected_results[i]

    # for query, expected_result in zip(test_queries, expected_results):
    #     result = ner.perform_ner(query)
    #     print(result)
    #     assert result == expected_result
