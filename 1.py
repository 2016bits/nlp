import re

s = "2_Hearts_-LRB-Kylie_Minogue_song-RRB-"

# 使用正则表达式去除-LRB-和-RRB-之间的所有元素
s_cleaned = re.sub(r'_-LRB-.*?-RRB-', '', s)

print(s_cleaned)
