# import pandas as pd
# import textdistance
from autocorrect import Speller


class Autocomplete:
    def __init__(self):
        # self.word_list = word_list
        self.speller = Speller()

    # def autocorrect(self, input_word):
    #     input_word = input_word.lower()
    #     if input_word in self.word_list:
    #         return "No error"  # Empty DataFrame
    #     else:
    #         try:
    #             return self.speller(input_word)
    #         except Exception as e:
    #             print(e)
    #             return "error"
    #         # sim_scores = [1 - textdistance.Jaccard(qval=2).similarity(word, input_word) for word in self.word_list]
    #         # suggestions = [word for _, word in sorted(zip(sim_scores, self.word_list), reverse=True)[:5]]
    #         # probs = [0.0] * 5  # Placeholder probabilities
    #         # similarities = [1 - score for score in sim_scores[:5]]  # Similarities based on Jaccard similarity
    #         # suggestions_df = pd.DataFrame({'Suggestion': suggestions, 'Probability': probs, 'Similarity': similarities})
    #         # return suggestions_df

    # def check_sentence(self, sentence):
    #     words = sentence.split()
    #     corrections = []
    #     hold = self.speller(sentence)
    #     for word in words:
    #         corrections.append(self.autocorrect(word))
    #     return hold
    
    def correct_spelling(self, input_text):
        # Split the input text into words
        words = input_text.split()

        # Correct each word
        corrected_words = [self.speller(word) for word in words]

        # Join the corrected words back into a single string
        corrected_text = ' '.join(corrected_words)

        return corrected_text
