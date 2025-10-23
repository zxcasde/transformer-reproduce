import os 
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HTTP_PROXY'] = 'http://10.126.191.254:7890'
os.environ['HTTPS_PROXY'] = 'http://10.126.191.254:7890'

from datasets import load_dataset

# 举例加载 “英语 -> 德语” (en-de) 的语言对
dataset = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-de", cache_dir='/data/yangguang/LLM/data')

# dataset = load_dataset("ag_news", cache_dir='/data/yangguang/LLM/data')

# 查看 train/validation/test 大小
print(dataset["train"].num_rows)
# print(dataset["validation"].num_rows)
print(dataset["test"].num_rows)

# 查看第一条数据
print(dataset["train"][0])
