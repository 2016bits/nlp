import sqlite3

replacements = {
    "-LRB-": "(",
    "-LSB-": "[",
    "-LCB-": "{",
    "-RCB-": "}",
    "-RRB-": ")",
    "-RSB-": "]",
    "-COLON-": ":",
}


def convert_evidence(evidence_list, c):
    evidence_text = ""
    for title, sent_index in evidence_list:
        sql = """select * from {} where id = "{}" ;""".format("documents", title)
        cursor = c.execute(sql)
        for row in cursor:
            lines = row[2].split('\n')
            for line in lines:
                sent_id = eval(line.split('\t')[0])
                if sent_id == sent_index:
                    sent_text = line.replace('{}\t'.format(sent_id), '')
                    sent_text = sent_text.replace('\t', ' ').replace('\n', ' ')
                    for replace in replacements:
                        if replace in sent_text:
                            sent_text = sent_text.replace(replace, replacements[replace])
                    evidence_text += " " + sent_text
    return evidence_text

def preprocess_dataset(dataset, db_path, sample_num):
    train_data = []
    max_len = 0
    
    sopport_num = 0
    refute_num = 0
    nei_num = 0

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    evidence = ""
    for data in dataset:
        if data['label'] == 'SUPPORTS' and sopport_num < sample_num:
            sopport_num += 1
            evidence = convert_evidence(data['evidence'], c)
            texts = f"{evidence}\nBased on the above information, is it true that {data['claim']}? true, false or uninformed? The answer is: "
            text_len = len(texts.split())
            if text_len > max_len:
                max_len = text_len

            train_data.append({
                'id': data['id'],
                'text': texts,
                'label': 'true'
            })

        elif data['label'] == 'REFUTES' and refute_num < sample_num:
            refute_num += 1
            evidence = convert_evidence(data['evidence'], c)
            texts = f"{evidence}\nBased on the above information, is it true that {data['claim']}? true, false or uninformed? The answer is: "
            text_len = len(texts.split())
            if text_len > max_len:
                max_len = text_len

            train_data.append({
                'id': data['id'],
                'text': texts,
                'label': 'false'
            })

        elif data['label'] == 'NOT ENOUGH INFO' and evidence and nei_num < sample_num:
            nei_num += 1
            texts = f"{evidence}\nBased on the above information, is it true that {data['claim']}? true, false or uninformed? The answer is: "
            text_len = len(texts.split())
            if text_len > max_len:
                max_len = text_len

            train_data.append({
                'id': data['id'],
                'text': texts,
                'label': 'uninformed'
            })
        elif sopport_num >= sample_num and refute_num >= sample_num and nei_num >= sample_num:
            break

    c.close()
    return train_data, max_len+5
