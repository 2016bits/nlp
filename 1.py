from unicodedata import normalize

# 原始字符串
page = "héllò"

# 使用 NFD 形式进行规范化
normalized_page = normalize("NFD", page)

# 输出结果
print(normalized_page)