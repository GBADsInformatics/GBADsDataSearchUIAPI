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

    def process_match_scores(self, species, category_to_check):
        original = species
        query_embed = self.embeddings_index[species]
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

    #     Query: Chinese bovine in 96
    # {'Names': -4.695542550086976, 'Places': 4.133115649223328, 'Colors': 1.6846088270346324, 'Species': 20.311160945892333} = species
    def extract_species(self, text):
        species = ""

        doc = self.nlp(text)
        # print(doc)

        # Extract species entity using NER
        for token in doc:
            # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            #     token.shape_, token.is_alpha, token.is_stop)
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                species = token.text

        species = species.lower()
        species = self.process_match_scores(species, "Species")
        return species

    def extract_country(self, text):
        country = ""
        newtext = text

        doc = self.nlp(text)

        # Extract country entity using NER
        for ent in doc.ents:
            if ent.label_ == "GPE" or ent.label_ == "LOC":
                country = ent.text

        # If nationality is mentioned, link it to the country using the nationality mapping
        nationality = self.link_nationality_to_country(text)
        if nationality and not country:
            newtext = newtext.replace(nationality, "")
            country = nationality
            return country, newtext

        newtext = newtext.replace(country, "")
        country = country.capitalize()
        return country, newtext

    def perform_ner(self, query):
        country, newtext = self.extract_country(query)
        cleaned_query = self.remove_stopwords(newtext)
        species = self.extract_species(cleaned_query)
        year = ", ".join(self.extract_years(query))

        # If year is still not identified, try to extract it from the remaining tokens
        if not year:
            year = ", ".join(
                self.extract_years(
                    " ".join(
                        [
                            token.text
                            for token in self.nlp(query)
                            if token.ent_type_ == ""
                        ]
                    )
                )
            )

        if species == "":
            species = None
        if year == "":
            year = None
        if country == "":
            country = None
        results = {"species": species, "year": year, "country": country}

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
