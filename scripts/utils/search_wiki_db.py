import sqlite3

db_path = "./data/Wikipedia/data/wikipedia.db"
db_table = "documents"
conn = sqlite3.connect(db_path)
c = conn.cursor()

titles = ["Greg_Robinson_-LRB-offensive_tackle-RRB-"]
sql = """select * from {} where id in ({seq})""".format(db_table, seq=','.join(['?'] * len(titles)))
cursor = c.execute(sql, titles)

for row in cursor:
    print(row[0])
    print(row[2])
