from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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
        custom_stop_words = ["The", "population"]

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

            highest_key = max(scores, key=scores.get)
            if highest_key == category_to_check:
                return original.capitalize()
            return None
        except:
            # Index error has occurred. This is likely due to the word not exsisting.
            return None

    #     Query: Chinese bovine in 96
    # {'Names': -4.695542550086976, 'Places': 4.133115649223328, 'Colors': 1.6846088270346324, 'Species': 20.311160945892333} = species
    def extract_species(self, text):
        species_list = []

        doc = self.nlp(text)
        for token in doc:
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                species = token.text.lower()
                species = self.process_match_scores(species, "Species")
                if species:
                    species_list.append(species.capitalize())

        return species_list

    def extract_country(self, text):
        country_list = []
        newtext = text
        doc = self.nlp(text)

        for ent in doc.ents:
            if ent.label_ == "GPE" or ent.label_ == "LOC":
                country = ent.text.capitalize()
                country_list.append(country)

        # If nationality is mentioned, link it to the country using the nationality mapping
        nationality = self.link_nationality_to_country(text)
        if nationality:
            newtext = newtext.replace(nationality, "")
            country_list.append(nationality.capitalize())

        real_countries = []

        for country in country_list:
            newtext = newtext.replace(country, "")
            country_in_ques = country.lower()
            if (country_in_ques.count(" ")==0):
                is_country = self.process_match_scores(country_in_ques, "Places")
                if is_country:
                    real_countries.append(country_in_ques.capitalize())
            else:
                real_countries.append(country_in_ques.capitalize())
        newtext = self.remove_stopwords(newtext)
        return real_countries, newtext

    def perform_ner(self, query):
        country, newtext = self.extract_country(query)
        cleaned_query = self.remove_stopwords(newtext)
        species = self.extract_species(cleaned_query)
        year = self.extract_years(query)

        if species == "":
            species = None
        if year == "":
            year = None
        if country == "":
            country = None
        results = {"species": species, "year": year, "country": country}
        print(results)
        return results

    def rank_years(self, yearInQues):
        yearInQues = yearInQues.lower()
        yearInQues = self.process_match_scores(yearInQues, "Years")
        return yearInQues

    def extract_years(self, text):
        years = []
        for token in self.nlp(text):
            ranked = self.rank_years(token.text)
            if ranked is not None:
                years.append(ranked)
            # if token.ent_type_ == "DATE" and token.text.isnumeric():
            #     years.append(token.text)
        
        return years
