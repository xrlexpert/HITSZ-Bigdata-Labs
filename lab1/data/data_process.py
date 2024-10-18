import jieba
from nltk.corpus import stopwords
import string

# 加载停用词列表（可以使用自己定义的停用词列表）
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file if line.strip()]  # 去掉每行的前后空白字符
    return set(stopwords)  # 返回一个集合以提高查找效率
stop_words = load_stopwords("D:\workspacefor_pyhton\BigDadaCourse\mySpider\mySpider\stopwords.txt")
def preprocess_text(text):
    # 使用 jieba 分词
    words = jieba.cut(text)
    # 去除停用词
    filtered_words = [word for word in words if word not in stop_words and word.strip() not in string.punctuation]
    return filtered_words

# 读取文本文件
with open("D:\workspacefor_pyhton\BigDadaCourse\mySpider\data.txt", 'r', encoding='utf-8') as file:
    text = file.read()

# 处理文本
processed_words = preprocess_text(text)
print(stop_words)

# 将处理后的词汇保存为txt文件
with open('D:\workspacefor_pyhton\BigDadaCourse\mySpider\processed_data.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(" ".join(processed_words))

print("处理后的文本已保存到 processed_text.txt 文件中。")
