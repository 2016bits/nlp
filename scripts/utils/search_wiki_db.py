import sqlite3
import json

def index2text(title, index):
    key = "{}_{}".format(title, index)
    return key

class Wiki:
    def __init__(self):
        db_path = "./data/Wikipedia/data/wikipedia.db"
        self.db_table = "documents"
        conn = sqlite3.connect(db_path)
        self.c = conn.cursor()
    
    def get_page(self, titles):
        sql = """select * from {} where id in ({seq})""".format(self.db_table, seq=','.join(['?'] * len(titles)))
        cursor = self.c.execute(sql, titles)
        results = {}
        for row in cursor:
            results[row[0]] = {
                "text": row[1],
                "line": row[2]
            }
        return results
    
    def get_sentence(self, sentences):
        titles = [sent for sent, _ in sentences]
        pages = self.get_page(titles)
        sent_dict = {}
        for title, index in sentences:
            page = pages[title]
            sentences = page['line'].split('\n')
            sent_id = eval(sentences[index].split('\t')[0])
            sent_text = sentences[index].replace('{}\t'.format(sent_id), '')
            key = index2text(title, index)
            if key not in sent_dict:
                sent_dict[key] = sent_text
        return sent_dict

class HopText:
    def __init__(self):
        data_path = "./results/analyze/FEVER_train_multi_hop.json"
    
        self.wiki = Wiki()
        with open(data_path, 'r') as f:
            self.dataset = json.load(f)

    def fun(self, top):
        hop_num = 0
        results = []
        for data in self.dataset:
            if data['hop'] == top and hop_num < 5:
                evidence_list = data['evidence']
                new_evidence_list = []
                sent_dict = self.wiki.get_sentence([evi for evidence in evidence_list for evi in evidence])
                for evidence in evidence_list:
                    new_evidence = []
                    for title, index in evidence:
                        text = sent_dict[index2text(title, index)]
                        new_evidence.append([title, index, text])
                    new_evidence_list.append(new_evidence)

                hop_num += 1
                results.append({
                    'id': data['id'],
                    'claim': data['claim'],
                    'label': data['label'],
                    'evidence': new_evidence_list,
                    'hop': data['hop']
                })
        return results

if __name__ == '__main__':
    hoptext = HopText()
    out_path = "test.json"
    results = []
    for top in [1, 2, 3, 4, 5]:
        results += hoptext.fun(top)
    with open(out_path, 'a+') as f1:
            f1.write(json.dumps(results, indent=2))

