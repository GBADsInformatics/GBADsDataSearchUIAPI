import spacy
import csv
import nltk
import numpy as np
import datetime
import requests
import time

from colorama import Fore, Back, Style
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    "Mistakes": ["rusia", "subsaharan", "saharan", "feeding"],
}


# Words -> category
categories = {word: key for key, words in data.items() for word in words}

# Load the whole embedding matrix
embeddings_index = {}
with open("glove.6B.200d.txt", encoding='utf-8') as f:
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
    # "Chickens in Great Britain between 2011 and 2010",
    # "Poultry in Canada in 2019",
    # "Canadian Poultry population",
    # "Ethiopia hens in 1997",
    # "Chinese bovine in 96",
    # "The population of cows in vietnam in the 2000s",
    # "Horses in 75",
    # "Random search that is nonsense",
    # "Poultry and Bovine population in Canada and Russia in 2011 and 2010",
    # "Bovine populaton in Canda and Rusia in 2009 or 2011",
    # "camel population in Subsaharan Africa",
    # "latest data on goats in Canada",
    # "cattle in South Africa this year",
    # "",
    # "Sheep and cattle in New Zealand in the 90s",
    # "Poultry and sheep in Australia in the 1980s",
    # "Cattle in Brazil between 1998 and 2002",
    # "Duck population in China in 2015",
    # "Swine in Germany in 2000",
    # "Elephants in Kenya in the 2010s",
    # "Buffalo in India in 85",
    # "Lion population in Africa",
    # "Birds and rabbits in the United States in 2018",
    # "Cat populaton in Italy and Spain in 2005 or 2006",
    # "Giraffe population in East Africa",
    # "latest data on horses in Mexico",
    # "dogs in Asia this year",

    # "Cats in Japan in 2015 and 2016",
    # "Dogs in France in 2020",
    # "Rabbits in the United Kingdom",
    # "Sheep population in New Zealand in the 80s",
    # "Goats in India between 2005 and 2010",
    # "Birds in the United States in the 21st century",
    # "Pigs in Germany in 1999",
    # "Tigers in India in 1990",
    # "Elephants in Thailand in the 2000s",
    # "Horses in Argentina in 2017",
    # "Lion population in South Africa",
    # "Ducks in Canada in 2014",
    # "Poultry in Russia in 2012",
    # "Camels in Saudi Arabia in 1995",
    # "Giraffes in Kenya in the 1990s",
    # "Buffalo in Nepal in 1988",
    # "Kangaroos in Australia",
    # "Monkeys in Brazil in 2007",
    # "Whales in Antarctica in 2021",
    # "Sharks in the Pacific Ocean",

    # "Pandas in China in 2010",
    # "Koalas in Australia in 2022",
    # "Penguins in Antarctica in 1995",
    # "Kangaroos in Australia in the 2000s",
    # "Turtles in Mexico in 2019",
    # "Bears in Canada in the 1980s",
    # "Lemurs in Madagascar in 2015, 2016, and 2017",
    # "Seals in The Arctic",
    # "Cheetahs in Namibia in 1998",
    # "Dolphins in The Bahamas in 2005",
    # "Puffins in Iceland",
    # "Otters in Canada in 2023",
    # # "Polar bears in The Arctic in the 1980s",
    # "Koalas in Australia in the 1990s",
    # "Giraffes in Kenya in 2020",
    # "Leopards in Africa",
    # "Pandas",
    # "2000",
    # # "The rainforest",
    # "Hawks in the United States in 2021",
    # # "Polar bears in The Arctic in the 2010s",
    # "Penguins in Antarctica in the 1990s",
    # # "Sharks in The Pacific Ocean in 2012",
    # "Crocodiles in Australia in 2001",
    # "Seals in Greenland",
    # "Kangaroos",

    # NEW NED TESTS
    "Snow Leopards in the Himalayas",
    "Guinea Pig population in North Carolina",
    "Black Rhinoceros in East Africa",
    "Behavioral studies on African Elephants",
    "Conservation efforts for Giant Pandas",
    "Migration patterns of Monarch Butterflies",
    "Habitat destruction and its impact on Orangutans",
    "The role of Wolves in ecosystems",
    "Genetic diversity in Bengal Tigers",
    "Threats to Sea Turtles nesting grounds",
    "The ecological importance of Bees",
    "Dietary preferences of Koalas",
    "Social structure of African Wild Dogs",
    "Adaptations of Camels to desert environments",
    "Breeding habits of Clownfish",
    "The significance of Coral Reefs",
    "Predation on Arctic Foxes by Polar Bears",
    "Behavioral differences between Domestic Cats and Wild Cats",
    "The impact of Climate Change on Penguins",
    "Reintroduction programs for California Condors",
]


def test_ner_accuracy():
    ner = PS.NER(
        nlp, data, categories, embeddings_index, data_embeddings, nationality_mapping
    )
    # Expected results for each test query

    current_year = datetime.datetime.now().year
    current_year_str = str(current_year)

    expected_results = [
        # {"species": ["Chickens"], "years": ["2011", "2010"], "countries": ["Great Britain"]},
        # {"species": ["Poultry"], "years": ["2019"], "countries": ["Canada"]},
        # {"species": ["Poultry"], "years": [], "countries": ["Canada"]},
        # {"species": ["Hens"], "years": ["1997"], "countries": ["Ethiopia"]},
        # {"species": ["Bovine"], "years": ["96"], "countries": ["China"]},
        # {"species": ["Cows"], "years": ["2000s"], "countries": ["Vietnam"]},
        # {"species": ["Horses"], "years": ["75"], "countries": []},
        # {"species": [], "years": [], "countries": []},
        # # Multiple speciies, years, and countries
        # {"species": ["Poultry", "Bovine"], "years": ["2011", "2010"], "countries": ["Canada", "Russia"]},
        # {"species": ["Bovine"], "years": ["2009", "2011"], "countries": []},
        # {"species": ["Camel"], "years": [], "countries": ["Subsaharan Africa"]},
        # {"species": ["Goats"], "years": [current_year_str], "countries": ["Canada"]},
        # {"species": ["Cattle"], "years": [current_year_str], "countries": ["South Africa"]},
        # {"species": [], "years": [], "countries": []},
        # {"species": ["Sheep", "Cattle"], "years": ["90s"], "countries": ["New Zealand"]},
        # {"species": ["Poultry", "Sheep"], "years": ["1980s"], "countries": ["Australia"]},
        # {"species": ["Cattle"], "years": ["1998", "2002"], "countries": ["Brazil"]},
        # {"species": ["Duck"], "years": ["2015"], "countries": ["China"]},
        # {"species": ["Swine"], "years": ["2000"], "countries": ["Germany"]},
        # {"species": ["Elephants"], "years": ["2010s"], "countries": ["Kenya"]},
        # {"species": ["Buffalo"], "years": ["85"], "countries": ["India"]},
        # {"species": ["Lion"], "years": [], "countries": ["Africa"]},
        # {"species": ["Birds", "Rabbits"], "years": ["2018"], "countries": ["The United States"]},
        # {"species": ["Cat"], "years": ["2005", "2006"], "countries": ["Italy", "Spain"]},
        # {"species": ["Giraffe"], "years": [], "countries": ["East Africa"]},
        # {"species": ["Horses"], "years": [current_year_str], "countries": ["Mexico"]},
        # {"species": ["Dogs"], "years": [current_year_str], "countries": ["Asia"]},

        # {"species": ["Cats"], "years": ["2015", "2016"], "countries": ["Japan"]},
        # {"species": ["Dogs"], "years": ["2020"], "countries": ["France"]},
        # {"species": ["Rabbits"], "years": [], "countries": ["The United Kingdom"]},
        # {"species": ["Sheep"], "years": ["80s"], "countries": ["New Zealand"]},
        # {"species": ["Goats"], "years": ["2005", "2010"], "countries": ["India"]},
        # {"species": ["Birds"], "years": [], "countries": ["The United States"]},
        # {"species": ["Pigs"], "years": ["1999"], "countries": ["Germany"]},
        # {"species": ["Tigers"], "years": ["1990"], "countries": ["India"]},
        # {"species": ["Elephants"], "years": ["2000s"], "countries": ["Thailand"]},
        # {"species": ["Horses"], "years": ["2017"], "countries": ["Argentina"]},
        # {"species": ["Lion"], "years": [], "countries": ["South Africa"]},
        # {"species": ["Ducks"], "years": ["2014"], "countries": ["Canada"]},
        # {"species": ["Poultry"], "years": ["2012"], "countries": ["Russia"]},
        # {"species": ["Camels"], "years": ["1995"], "countries": ["Saudi Arabia"]},
        # {"species": ["Giraffes"], "years": ["1990s"], "countries": ["Kenya"]},
        # {"species": ["Buffalo"], "years": ["1988"], "countries": ["Nepal"]},
        # {"species": ["Kangaroos"], "years": [], "countries": ["Australia"]},
        # {"species": ["Monkeys"], "years": ["2007"], "countries": ["Brazil"]},
        # {"species": ["Whales"], "years": ["2021"], "countries": ["Antarctica"]},
        # {"species": ["Sharks"], "years": [], "countries": ["The Pacific Ocean"]},

        # {"species": ["Pandas"], "years": ["2010"], "countries": ["China"]},
        # {"species": ["Koalas"], "years": ["2022"], "countries": ["Australia"]},
        # {"species": ["Penguins"], "years": ["1995"], "countries": ["Antarctica"]},
        # {"species": ["Kangaroos"], "years": ["2000s"], "countries": ["Australia"]},
        # {"species": ["Turtles"], "years": ["2019"], "countries": ["Mexico"]},
        # {"species": ["Bears"], "years": ["1980s"], "countries": ["Canada"]},
        # {"species": ["Lemurs"], "years": ["2015", "2016", "2017"], "countries": ["Madagascar"]},
        # {"species": ["Seals"], "years": [], "countries": ["Arctic"]},
        # {"species": ["Cheetahs"], "years": ["1998"], "countries": ["Namibia"]},
        # {"species": ["Dolphins"], "years": ["2005"], "countries": ["Bahamas"]},
        # {"species": ["Puffins"], "years": [], "countries": ["Iceland"]},
        # {"species": ["Otters"], "years": ["2023"], "countries": ["Canada"]},
        # # {"species": ["Polar bears"], "years": ["1980s"], "countries": ["Arctic"]},
        # {"species": ["Koalas"], "years": ["1990s"], "countries": ["Australia"]},
        # {"species": ["Giraffes"], "years": ["2020"], "countries": ["Kenya"]},
        # {"species": ["Leopards"], "years": [], "countries": ["Africa"]},
        # {"species": ["Pandas"], "years": [], "countries": []},
        # {"species": [], "years": ["2000"], "countries": []},
        # # {"species": [], "years": [], "countries": ["The rainforest"]},
        # {"species": ["Hawks"], "years": ["2021"], "countries": ["The United States"]},
        # # {"species": ["Polar bears"], "years": ["2010s"], "countries": ["Arctic"]},
        # {"species": ["Penguins"], "years": ["1990s"], "countries": ["Antarctica"]},
        # # {"species": ["Sharks"], "years": ["2012"], "countries": ["The Pacific Ocean"]},
        # {"species": ["Crocodiles"], "years": ["2001"], "countries": ["Australia"]},
        # {"species": ["Seals"], "years": [], "countries": ["Greenland"]},
        # {"species": ["Kangaroos"], "years": [], "countries": []},

        # NEW NED RESULTS
        {"species": ["Snow Leopards"], "years": [], "countries": ["Himalayas"]},
        {"species": ["Guinea Pig"], "years": [], "countries": ["North Carolina"]},
        {"species": ["Black Rhinoceros"], "years": [], "countries": ["East Africa"]},
        {"species": ["African Elephants"], "years": [], "countries": []},
        {"species": ["Giant Pandas"], "years": [], "countries": []},
        {"species": ["Monarch Butterflies"], "years": [], "countries": []},
        {"species": ["Orangutans"], "years": [], "countries": []},
        {"species": ["Wolves"], "years": [], "countries": []},
        {"species": ["Bengal Tigers"], "years": [], "countries": []},
        {"species": ["Sea Turtles"], "years": [], "countries": []},
        {"species": ["Bees"], "years": [], "countries": []},
        {"species": ["Koalas"], "years": [], "countries": []},
        {"species": ["African Wild Dogs"], "years": [], "countries": []},
        {"species": ["Camels"], "years": [], "countries": []},
        {"species": ["Clownfish"], "years": [], "countries": []},
        {"species": ["Coral Reefs"], "years": [], "countries": []},
        {"species": ["Arctic Foxes", "Polar Bears"], "years": [], "countries": []},
        {"species": ["Domestic Cats", "Wild Cats"], "years": [], "countries": []},
        {"species": ["Penguins"], "years": [], "countries": []},
        {"species": ["California Condors"], "years": [], "countries": ["California"]},
    ]

    for i in range(0, len(test_queries)):
        result = ner.perform_ner(test_queries[i])
        print(Fore.CYAN + f'{result}')
        print(Fore.MAGENTA + f'{expected_results[i]}')
        print(Fore.GREEN + "PASS" if result == expected_results[i] else Fore.RED + "FAIL")
        assert result == expected_results[i]

    # for query, expected_result in zip(test_queries, expected_results):
    #     result = ner.perform_ner(query)
    #     print(result)
    #     assert result == expected_result


def make_ner_api_call(query):
    url = "https://www.gbadske.org/search/api/search"
    params = {'query': query}

    try:
        start_time = time.time()  # Record the start time before making the API call
        response = requests.get(url, params=params)
        end_time = time.time()  # Record the end time after receiving the API response
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Calculate the response time
            response_time = end_time - start_time
            return response_time
        else:
            return -1
    except requests.RequestException as e:
        print(f"Query: {query}, Error making API request:", e)


def test_ner_performance():
    api_errors = 0
    total_response_time = 0
    for query in test_queries:
        result = make_ner_api_call(query)
        if result == -1:
            api_errors += 1
        else:
            total_response_time += result
    average_response_time = total_response_time / (len(test_queries) - api_errors)
    print(average_response_time)
