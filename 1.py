import re

s = "evidences = Find_evidences(wiki_pages, \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT\", \"NXT"
num = len(re.findall(r'"', s))
print(num)
if num % 2:
    s += '"'
print(s)
