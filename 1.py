import sqlite3

conn = sqlite3.connect('./data/Wikipedia/data/wikipedia.db')
c = conn.cursor()

title = "Boston_Celtics"
line_id = 3
sql = """select * from documents where id = "{}";""".format(title)

cursor = c.execute(sql)
text = ""
for row in cursor:
    lines = row[2]
    lines = row[2].split('\n')
    for line in lines:
        sent_id = eval(line.split('\t')[0])
        if sent_id == line_id:
            sent_text = line.replace('{}\t'.format(sent_id), '')
            text += sent_text
print(text)
