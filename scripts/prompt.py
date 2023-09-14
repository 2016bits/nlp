FEVER_PROGRAM_FC = """# FEVER_PROGRAM_FC = ''''Generate a python-like fact_checking that describes the reasoning steps required to verify the claim step-by-step. You can call three functions in the program: 1. Search_pages() to search relevant wikipedia pages；2. Find_evidences() to find relevant evidences; 3. Verify() to verify the claim. Several examples are given as follows.

# The claim is that The Boston Celtics play their home games at TD Garden.
def fact_checking():
    claim = "The Boston Celtics play their home games at TD Garden ."
    wiki_pages = Search_pages("Boston Celtics", "TD Garden")
    evidences = Find_evidences(wiki_pages, "Celtics", "home", "TD Garden")
    label = Verify(claim, evidences)

# The claim is that Roman Atwood is a content creator .
def fact_checking():
    claim = "Roman Atwood is a content creator ."
    wiki_pages = Search_pages("Roman Atwood")
    evidences = Find_evidences(wiki_pages, "create")
    label = Verify(claim, evidences)

# The claim is that Adrienne Bailon is an accountant.
def fact_checking():
    claim = "Adrienne Bailon is an accountant."
    wiki_pages = Search_pages("Adrienne Bailon")
    evidences = Find_evidences(wiki_pages, "account")
    label = Verify(claim, evidences)

# The claim is that The Ten Commandments is an epic film.
def fact_checking():
    claim = "The Ten Commandments is an epic film."
    wiki_pages = Search_pages("Ten Commandments")
    evidences = Find_evidences(wiki_pages, "epic", "film")
    label = Verify(claim, evidences)

# The claim is that There is a movie called The Hunger Games.
def fact_checking():
    claim = "There is a movie called The Hunger Games."
    wiki_pages = Search_pages("Hunger Games")
    evidences = Find_evidences(wiki_pages, "movie")
    label = Verify(claim, evidences)

# The claim is that Puerto Rico is not an unincorporated territory of the United States .
def fact_checking():
    claim = "Puerto Rico is not an unincorporated territory of the United States ."
    wiki_pages = Search_pages("Puerto Rico")
    evidences = Find_evidences(wiki_pages, "unincorporated territory", "United States")
    label = Verify(claim, evidences)

# The claim is that Michael Giacchino composed the score for Doctor Strange.
def fact_checking():
    claim = "Michael Giacchino composed the score for Doctor Strange."
    wiki_pages = Search_pages("Michael Giacchino", "Doctor Strange")
    evidences = Find_evidences(wiki_pages, "Michael Giacchino", "compose", "Doctor Strange")
    label = Verify(claim, evidences)

# The claim is that Robert J. O'Neill was born April 10, 1976
def fact_checking():
    claim = "Robert J. O'Neill was born April 10, 1976 ."
    wiki_pages = Search_pages("Robert J. O'Neill")
    evidences = Find_evidences(wiki_pages, "born", "April 10", "1976")
    label = Verify(claim, evidences)

# The claim is that Peggy Sue Got Married is a Egyptian film released in 1986.
def fact_checking():
    claim = "Peggy Sue Got Married is a Egyptian film released in 1986."
    wiki_pages = Search_pages("Peggy Sue Got Married")
    evidences = Find_evidences(wiki_pages, "Egypt", "film", "release", "1986")
    label = Verify(claim, evidences)

# The claim is that Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
def fact_checking():
    claim = "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
    wiki_pages = Search_pages("Nikolaj Coster-Waldau")
    evidences = Find_evidences(wiki_pages, "Nikolaj", "Fox television")
    label = Verify(claim, evidences)

# The claim is that [[CLAIM]]
def fact_checking():"""

SCIFACT_PROGRAM_FC = """Generate a python-like program that describes the reasoning steps required to verify the claim step-by-step. You can call three functions in the program: 1. 

# The claim is that 
def program():

    label = Predict()

# The claim is that [[CLAIM]]
def program():"""


class Prompt_Loader:
    def __init__(self):
        self.fever_program_fc = FEVER_PROGRAM_FC
        self.scifact_program_fc = SCIFACT_PROGRAM_FC

    def prompt_construction(self, claim, dataset_name):
        template = None
        if dataset_name == 'FEVER':
            template = self.fever_program_fc
        elif dataset_name == 'SCIFACT':
            template = self.scifact_program_fc
        else:
            raise NotImplementedError
        
        return template.replace('[[CLAIM]]', claim)
