import re
import nltk
from allennlp.predictors import Predictor
from unicodedata import normalize

class Doc_Retrieval:
    def __init__(self, k_wiki_results=None):
        self.k_wiki_results = k_wiki_results
        self.proter_stemm = nltk.PorterStemmer()
        self.tokenizer = nltk.word_tokenize
        self.predictor = Predictor.from_path("/data/yangjun/tools/elmo-constituency-parser-2020.02.10.tar.gz")

    def get_NP(self, tree, nps):
        if isinstance(tree, dict):
            if "children" not in tree:
                if tree['nodeType'] == "NP":
                    nps.append(tree['word'])
            elif "children" in tree:
                if tree['nodeType'] == "NP":
                    nps.append(tree['word'])
                    self.get_NP(tree['children'], nps)
                else:
                    self.get_NP(tree['children'], nps)
        elif isinstance(tree, list):
            for sub_tree in tree:
                self.get_NP(sub_tree, nps)

        return nps

    def get_subjects(self, tree):
        subject_words = []
        subjects = []
        for subtree in tree['children']:
            if subtree['nodeType'] == "VP" or subtree['nodeType'] == 'S' or subtree['nodeType'] == 'VBZ':
                subjects.append(' '.join(subject_words))
                subject_words.append(subtree['word'])
            else:
                subject_words.append(subtree['word'])
        return subjects

    def get_noun_phrases(self, claim):
        tokens = self.predictor.predict(claim)
        nps = []
        tree = tokens['hierplane_tree']['root']
        noun_phrases = self.get_NP(tree, nps)
        subjects = self.get_subjects(tree)
        for subject in subjects:
            if len(subject) > 0:
                noun_phrases.append(subject)
        return list(set(noun_phrases))


claim = "The New Jersey Turnpike has zero shoulders."
model = Doc_Retrieval()
noun_phrases = model.get_noun_phrases(claim)
print(noun_phrases)
# >>> ['zero shoulders', 'The New Jersey Turnpike']