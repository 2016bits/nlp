import sqlite3
import torch
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

class Search_wikipage:
    """根据文档标题从Wikipedia中获取相应页面"""
    def __init__(self, db_path, db_table):
        self.db_table = db_table
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()
    
    def select_wikipage(self, titles):
        results = {}
        titles = [title.replace(' ', '_') for title in titles]

        # retrieve database to get abstract
        sql = """select * from {} where id in ({seq})""".format(self.db_table, seq=','.join(['?'] * len(titles)))
        cursor = self.c.execute(sql, titles)
        for row in cursor:
            # key: id, value: lines
            results[row[0]] = row[2]
        return results

class Select_sentence:
    """从页面中选择证据句子"""
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
    
    def calculate_sentence_score(self, text1, text2):
        input_ids = self.tokenizer(text1, text2, return_tensors="pt", truncation=True).to(self.device)
        output = self.model(input_ids['input_ids'])
        prediction = torch.softmax(output["logits"][0], -1)
        prob, _ = torch.max(prediction, dim=-1)
        return prob
    
    def select_sentence(self, claim, pages, key_words):
        if pages == {}:
            return {
                'evidence_ids': [],
                'evidence_texts': []
            }
        
        # evidence_ids is for calculating predicted evidence results, evidence_texts is for generating verified results
        evidence_ids = []
        evidence_texts = []

        # 先找到含有关键词的句子
        sent_list = []
        for title, page in pages.items():
            sentences = page.split('\n')
            for sent in sentences:
                if sent and sent[0].isdigit:
                    # find keyword in the sentence
                    sent_chunck = sent.split('\t')
                    sent_id = eval(sent_chunck[0])
                    if len(sent_chunck) > 1 and len(sent_chunck[1]) > 0:
                        sent_text = sent_chunck[1]
                        score = self.calculate_sentence_score(claim, sent_text)
                        sent_list.append({
                            'id': [title, sent_id],
                            'text': sent_text,
                            'score': score
                        })
                            
        # for title, page in pages.items():
        #     sentences = page.split('\n')
        #     for sent in sentences:
        #         for word in key_words:
        #             if sent.lower().find(word.lower()) != -1:
        #                 # find keyword in the sentence
        #                 sent_chunck = sent.split('\t')
        #                 sent_id = eval(sent_chunck[0])
        #                 if len(sent_chunck) > 1 and len(sent_chunck[1]) > 0:
        #                     sent_text = sent_chunck[1]
        #                     score = self.calculate_sentence_score(claim, sent_text)
        #                     sent_list.append({
        #                         'id': [title, sent_id],
        #                         'text': sent_text,
        #                         'score': score
        #                     })
                            
        #                     break
                
        # 对sent_list按照score值进行排序
        sorted_list = sorted(sent_list, key=lambda x: x['score'], reverse=True)
        for index in range(min(len(sorted_list), 5)):
            evidence_ids.append(sorted_list[index]['id'])
            evidence_texts.append(sorted_list[index]['text'])

        # 对含有关键词的句子，使用encoder计算其与claim的语义相似性，选择top5作为最终的证据句子
        return {
            'evidence_ids': evidence_ids,
            'evidence_texts': evidence_texts
        }