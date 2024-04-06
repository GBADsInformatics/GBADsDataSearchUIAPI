import datetime
from gliner import GLiNER
from colorama import Fore, Back, Style

model = GLiNER.from_pretrained("urchade/gliner_base")

text = """
Camels in Russia in 2019 and Horses in China in 2012
"""

labels = ["species", "years", "countries", "continent", "country"]

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

current_year = datetime.datetime.now().year
current_year_str = str(current_year)

expected_results = [
    {"species": ["Chickens"], "years": ["2011", "2010"], "countries": ["Great Britain"]},
    {"species": ["Poultry"], "years": ["2019"], "countries": ["Canada"]},
    {"species": ["Poultry"], "years": [], "countries": ["Canada"]},
    {"species": ["Hens"], "years": ["1997"], "countries": ["Ethiopia"]},
    {"species": ["Bovine"], "years": ["96"], "countries": ["China"]},
    {"species": ["Cows"], "years": ["2000s"], "countries": ["Vietnam"]},
    {"species": ["Horses"], "years": ["75"], "countries": []},
    {"species": [], "years": [], "countries": []},
    # Multiple speciies, years, and countries
    {"species": ["Poultry", "Bovine"], "years": ["2011", "2010"], "countries": ["Canada", "Russia"]},
    {"species": ["Bovine"], "years": ["2009", "2011"], "countries": []},
    {"species": ["Camel"], "years": [], "countries": ["Subsaharan africa"]},
    {"species": ["Goats"], "years": [current_year_str], "countries": ["Canada"]},
    {"species": ["Cattle"], "years": [current_year_str], "countries": ["South Africa"]},
    {"species": [], "years": [], "countries": []},
    {"species": ["Sheep", "Cattle"], "years": ["90s"], "countries": ["New Zealand"]},
    {"species": ["Poultry", "Sheep"], "years": ["1980s"], "countries": ["Australia"]},
    {"species": ["Cattle"], "years": ["1998", "2002"], "countries": ["Brazil"]},
    {"species": ["Duck"], "years": ["2015"], "countries": ["China"]},
    {"species": ["Swine"], "years": ["2000"], "countries": ["Germany"]},
    {"species": ["Elephants"], "years": ["2010s"], "countries": ["Kenya"]},
    {"species": ["Buffalo"], "years": ["85"], "countries": ["India"]},
    {"species": ["Lion"], "years": [], "countries": ["Africa"]},
    {"species": ["Birds", "Rabbits"], "years": ["2018"], "countries": ["United States"]},
    {"species": ["Cat"], "years": ["2005", "2006"], "countries": ["Italy", "Spain"]},
    {"species": ["Giraffe"], "years": [], "countries": ["East Africa"]},
    {"species": ["Horses"], "years": [current_year_str], "countries": ["Mexico"]},
    {"species": ["Dogs"], "years": [current_year_str], "countries": ["Asia"]},

    {"species": ["Cats"], "years": ["2015", "2016"], "countries": ["Japan"]},
    {"species": ["Dogs"], "years": ["2020"], "countries": ["France"]},
    {"species": ["Rabbits"], "years": [], "countries": ["United Kingdom"]},
    {"species": ["Sheep"], "years": ["80s"], "countries": ["New Zealand"]},
    {"species": ["Goats"], "years": ["2005", "2010"], "countries": ["India"]},
    {"species": ["Birds"], "years": [], "countries": ["United States"]},
    {"species": ["Pigs"], "years": ["1999"], "countries": ["Germany"]},
    {"species": ["Tigers"], "years": ["1990"], "countries": ["India"]},
    {"species": ["Elephants"], "years": ["2000s"], "countries": ["Thailand"]},
    {"species": ["Horses"], "years": ["2017"], "countries": ["Argentina"]},
    {"species": ["Lion"], "years": [], "countries": ["South Africa"]},
    {"species": ["Ducks"], "years": ["2014"], "countries": ["Canada"]},
    {"species": ["Poultry"], "years": ["2012"], "countries": ["Russia"]},
    {"species": ["Camels"], "years": ["1995"], "countries": ["Saudi Arabia"]},
    {"species": ["Giraffes"], "years": ["1990s"], "countries": ["Kenya"]},
    {"species": ["Buffalo"], "years": ["1988"], "countries": ["Nepal"]},
    {"species": ["Kangaroos"], "years": [], "countries": ["Australia"]},
    {"species": ["Monkeys"], "years": ["2007"], "countries": ["Brazil"]},
    {"species": ["Whales"], "years": ["2021"], "countries": ["Antarctica"]},
    {"species": ["Sharks"], "years": [], "countries": ["Pacific Ocean"]},

    {"species": ["Pandas"], "years": ["2010"], "countries": ["China"]},
    {"species": ["Koalas"], "years": ["2022"], "countries": ["Australia"]},
    {"species": ["Penguins"], "years": ["1995"], "countries": ["Antarctica"]},
    {"species": ["Kangaroos"], "years": ["2000s"], "countries": ["Australia"]},
    {"species": ["Turtles"], "years": ["2019"], "countries": ["Mexico"]},
    {"species": ["Bears"], "years": ["1980s"], "countries": ["Canada"]},
    {"species": ["Lemurs"], "years": ["2015", "2016", "2017"], "countries": ["Madagascar"]},
    {"species": ["Seals"], "years": [], "countries": ["Arctic"]},
    {"species": ["Cheetahs"], "years": ["1998"], "countries": ["Namibia"]},
    {"species": ["Dolphins"], "years": ["2005"], "countries": ["Bahamas"]},
    {"species": ["Puffins"], "years": [], "countries": ["Iceland"]},
    {"species": ["Otters"], "years": ["2023"], "countries": ["Canada"]},
    # {"species": ["Polar bears"], "years": ["1980s"], "countries": ["Arctic"]},
    {"species": ["Koalas"], "years": ["1990s"], "countries": ["Australia"]},
    {"species": ["Giraffes"], "years": ["2020"], "countries": ["Kenya"]},
    {"species": ["Leopards"], "years": [], "countries": ["Africa"]},
    {"species": ["Pandas"], "years": [], "countries": []},
    {"species": [], "years": ["2000"], "countries": []},
    # {"species": [], "years": [], "countries": ["The rainforest"]},
    {"species": ["Hawks"], "years": ["2021"], "countries": ["United States"]},
    # {"species": ["Polar bears"], "years": ["2010s"], "countries": ["Arctic"]},
    {"species": ["Penguins"], "years": ["1990s"], "countries": ["Antarctica"]},
    # {"species": ["Sharks"], "years": ["2012"], "countries": ["The Pacific Ocean"]},
    {"species": ["Crocodiles"], "years": ["2001"], "countries": ["Australia"]},
    {"species": ["Seals"], "years": [], "countries": ["Greenland"]},
    {"species": ["Kangaroos"], "years": [], "countries": []},
]

def predict_ents(text):
    try:
        entities = model.predict_entities(text, labels, threshold=0.5)
        species = []
        countries = []
        years = []
        for entity in entities:
            if (entity["label"] == "years"):
                years.append(entity["text"].capitalize())
            if (entity["label"] == "species"):
                species.append(entity["text"].title())
            if (entity["label"] == "countries" or entity["label"] == "continent" or entity["label"] == "country"):
                # Need to remove 'The' if it's included in the name of the country/continent
                remove_random_words = entity["text"].lower().split(' ')
                if 'the' in remove_random_words:
                    remove_random_words.remove('the')
                remove_random_words = ' '.join(remove_random_words)
                countries.append(remove_random_words.title())
            print(Fore.MAGENTA + entity["label"] + ' -> ' + entity["text"])
        result = {"species": species, "years": years, "countries": countries}
        print(Fore.CYAN + f'{result}')
        return result
    except IndexError:
        return ""


count = 0
passed_tests = 0
failed_tests = []
for test in test_queries:
    print(Fore.WHITE + test)
    if(expected_results[count] == predict_ents(test)):
        passed_tests = passed_tests + 1
        print(Fore.GREEN + 'Passed')
    else:
        failed_tests.append(count)
        print(Fore.RED + 'FAILED')
        print(Fore.YELLOW + f'Expected: {expected_results[count]}')
    print(Style.RESET_ALL)
    count = count + 1

print(f"Passed {passed_tests}/{len(expected_results)}")
print(f"Failed: {failed_tests}")

