import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# import Autocomplete
import nltk
import requests
nltk.download("stopwords")
nltk.download("punkt")

class NER:
    def __init__(
        self,
        nlp,
        data,
        categories,
        embeddings_index,
        data_embeddings,
        nationality_mapping,
    ):
        self.nlp = nlp
        self.data = data
        self.categories = categories
        self.embeddings_index = embeddings_index
        self.data_embeddings = data_embeddings
        self.nationality_mapping = nationality_mapping
        # self.auto_complete = Autocomplete.Autocomplete()

    # Function to link nationality to country
    def link_nationality_to_country(self, text):
        country = ""
        for token in self.nlp(text):
            if token.text.lower() in self.nationality_mapping:
                country = self.nationality_mapping[token.text.lower()]
                break
        return country

    # Function to remove stopwords from text
    def remove_stopwords(self, text):
        all_stopwords = stopwords.words("english")

        # Custom stopwords
        custom_stop_words = ["The", "population", "the"]

        for word in custom_stop_words:
            all_stopwords.append(word)

        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if word not in all_stopwords]
        return " ".join(tokens_without_sw)

    # check_meaning -> Whatever word is to be checked
    # category_to_check -> category to check against it
    def process_match_scores(self, check_meaning, category_to_check):
        try:
            original = check_meaning
            query_embed = self.embeddings_index[check_meaning]
            scores = {}
            for word, embed in self.data_embeddings.items():
                category = self.categories[word]
                dist = query_embed.dot(embed)
                dist /= len(self.data[category])
                scores[category] = scores.get(category, 0) + dist

            # print("LOOKING FOR: " + category_to_check)
            # print(original)
            # print(scores)
            print(check_meaning.upper())
            print(scores)
            highest_key = max(scores, key=scores.get)
            print(highest_key)
            if (category_to_check == "Places"):
                if (highest_key == category_to_check or highest_key == "Continents" or highest_key == "Regions"):
                    return original.title()
            if highest_key == category_to_check:
                return original.title()
            return None
        except Exception as e:
            # Index error has occurred. This is likely due to the word not exsisting.
            # print(e)
            print(f"An error has occurred in process_match_scores(). ERROR: {e}")
            return None

    #     Query: Chinese bovine in 96
    # {'Names': -4.695542550086976, 'Places': 4.133115649223328, 'Colors': 1.6846088270346324, 'Species': 20.311160945892333} = species
    def extract_species(self, text):
        species_list = []

        doc = self.nlp(text)
        for token in doc:
            # print("TTOKEN: " + token.text)
            # print("POS: " + str(token.pos_))
            species = token.text.lower()
            species = self.process_match_scores(species, "Species")
            if species:
                species_list.append(species.title())

        return species_list

    def extract_country(self, text):
        country_list = []
        newtext = text
        doc = self.nlp(text)

        for ent in doc.ents:
            if ent.label_ == "GPE" or ent.label_ == "LOC":
                country = ent.text.title()
                country_list.append(country)

        # If nationality is mentioned, link it to the country using the nationality mapping
        nationality = self.link_nationality_to_country(text)
        if nationality:
            newtext = newtext.replace(nationality, "")
            country_list.append(nationality.title())

        real_countries = []

        for country in country_list:
            newtext = newtext.replace(country, "")
            country_in_ques = country.lower()
            if (country_in_ques.count(" ") == 0):
                is_country = self.process_match_scores(country_in_ques, "Places")
                if is_country:
                    real_countries.append(country_in_ques.title())
            else:
                real_countries.append(country_in_ques.title())
        newtext = self.remove_stopwords(newtext)
        return real_countries, newtext
    

    # def derive_label(self, text):
    #     print("RETURNED FROM DERIVE LABEL")
    #     is_country = self.process_match_scores(text, "Places")
    #     is_species = self.process_match_scores(text, "Species")
    #     print(is_country)
    #     print(is_species)
    #     if is_country:
    #         return "Place"
    #     if is_species:
    #         return "Species"
    
    def ned_with_dbpedia_spotlight_verifier(self, text):
        url = "http://api.dbpedia-spotlight.org/en/annotate"
        params = {
            "text": text,
            "confidence": 0.5,
            "support": 20
        }
        headers = {
            "Accept": "application/json"
        }
        response = requests.get(url, params=params, headers=headers)
        try:
            response.raise_for_status()
            annotations = response.json()
            return annotations
        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
            print(f"Response content: {response.content}")
            return None
        except requests.exceptions.JSONDecodeError as err:
            print(f"JSON decoding error occurred: {err}")
            print(f"Response content: {response.content}")
            return None

    def perform_ner(self, query):
        country, newtext = self.extract_country(query)
        # cleaned_query = self.remove_stopwords(newtext)
        # print("CLEANED: " + cleaned_query)
        species = self.extract_species(query)
        year = self.extract_years(query)

        # Add verification step after we've derived our categories
        try:
            verified_list_results = self.ned_with_dbpedia_spotlight_verifier(query)
            print("RETURNED FROM TEST FUNCTION")
            verified_species = []
            surface_forms = []
            if verified_list_results:
                surface_forms = [resource['@surfaceForm'] for resource in verified_list_results['Resources']]
                print(surface_forms)
                # for any_ent in surface_forms:
                #     check_label = self.derive_label(any_ent)
                #     if (check_label == "Species"):
                #         verified_species.append(any_ent)
            # print(verified_species)
            # if len(verified_species) > 0:
            for spec_ent in surface_forms:
                for val in species:
                    if (val in spec_ent):
                        species[species.index(val)] = spec_ent
        except KeyError as e:
            print(f"Verification Step failed. Error: {e}")


        if species == "":
            species = None
        if year == "":
            year = None
        if country == "":
            country = None
        results = {"species": species, "years": year, "countries": country}
        # print(results)
        return results

    def rank_years(self, yearInQues):
        yearInQues = yearInQues.lower()
        yearInQues = self.process_match_scores(yearInQues, "Years")
        return yearInQues

    def is_convertible_to_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def extract_years(self, text):
        years = []
        for token in self.nlp(text):
            ranked = self.rank_years(token.text)
            # Checks if the string can be converted into a number
            the_text = str(token.text)
            check_convert = self.is_convertible_to_number(the_text)
            if (ranked is not None) or (check_convert):
                years.append(the_text)

        check_curr_year = self.find_curr_year(text)

        if (check_curr_year not in years) and (check_curr_year != ""):
            years.append(check_curr_year)

        try:
            # Years that are verified by checking if it's a digit or not.
            washed_years = []

            for year in years:
                y = year[:-1]  # Get the part of the string without the last character
                x = year[-1]   # Get the last character
                if year.isdigit():
                    washed_years.append(year)
                elif y.isdigit() and x == 's':
                    washed_years.append(year)
            return washed_years
        except Exception as e:
            # print("extract_years: " + str(e))
            print(f"An error has occurred in extract_years(). ERROR: {e}")
            return years

    # Checks for specific keywords within a sentence to determine if asked about the current year
    def find_curr_year(self, text):
        # Convert the sentence to lowercase for case-insensitive search
        lowercase_sentence = text.lower()

        # List of keywords to search for
        keywords = ["this year", "latest", "current"]

        # Search for keywords in the lowercase sentence
        found_keywords = [keyword for keyword in keywords if keyword in lowercase_sentence]

        if found_keywords:
            current_year = datetime.datetime.now().year
            current_year_str = str(current_year)
            return current_year_str
        else:
            return ""
