import sqlite3
from drqa.retriever import DocDB, utils

try:
    from utils.retriever import TopNDocsTopNSents
except:
    from scripts.utils.retriever import TopNDocsTopNSents


class FeverDocDB(DocDB):

    def __init__(self,path=None):
        super().__init__(path)

    def get_doc_lines(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT lines FROM documents WHERE id = ?",
            (utils.normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def get_non_empty_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents WHERE length(trim(text)) > 0")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

class Search_Tfidf:
    def __init__(self, db_path, max_page, max_sent, tfidf_model):
        self.db = FeverDocDB(db_path)
        self.method = TopNDocsTopNSents(self.db, max_page, max_sent, tfidf_model)
    
    def search_sents(self, claim):
        sent_ids, sent_texts = self.method.get_sentences_for_claim(claim)
        return {
            'evidence_ids': sent_ids,
            'evidence_texts': sent_texts
        }

class Search_Wiki:
    def __init__(self, db_path, db_table, max_evidence_num):
        self.db_path = db_path
        self.db_table = db_table
        self.max_evidence_num = max_evidence_num
    
    def search_wikipages(self, titles):
        # search wikipedia pages which title is in titles
        # connect to database
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        results = {}
        titles = [title.replace(' ', '_') for title in titles]

        # retrieve database to get abstract
        sql = """select * from {} where id in ({seq})""".format(self.db_table, seq=','.join(['?'] * len(titles)))
        cursor = c.execute(sql, titles)
        for row in cursor:
            # key: id, value: lines
            results[row[0]] = row[2]
        return results
    
    def find_evidences(self, wiki_pages, key_words):
        # find evidence sentences in wikipedia pages according to key words
        if wiki_pages == {}:
            return {
                'evidence_ids': [],
                'evidence_texts': []
            }
        
        # evidence_ids is for calculating predicted evidence results, evidence_texts is for generating verified results
        evidence_ids = []
        evidence_texts = []
        for title, page in wiki_pages.items():
            sentences = page.split('\n')
            for sent in sentences:
                flag = True
                for word in key_words:
                    if sent.lower().find(word.lower()) == -1:
                        # do not find key words in the sentence
                        flag = False
                        break
                if flag:
                    sent_id = eval(sent.split('\t')[0])
                    sent_text = sent.replace('{}\t'.format(sent_id), '')
                    evidence_ids.append([title, sent_id])
                    evidence_texts.append(sent_text)
        return {
            'evidence_ids': evidence_ids,
            'evidence_texts': evidence_texts
        }
