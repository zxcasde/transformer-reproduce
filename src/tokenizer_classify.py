from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os

# 准备训练数据
def get_training_corpus(dataset):
    for i in range(0, len(dataset['train']), 1000):  # 分批处理避免内存问题
        yield dataset['train'][i:i+1000]['text']

def get_tokenizer():
    tokenizer_src = Tokenizer.from_file("./ag_news_tokenizer/tokenizer.json")

    return tokenizer_src

def get_data():
    dataset = load_dataset("ag_news", cache_dir='/data/yangguang/LLM/data')

    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    return (train_texts, train_labels), (test_texts, test_labels)

if __name__=='__main__':
    # 加载数据集
    dataset = load_dataset("ag_news", cache_dir='/data/yangguang/LLM/data')

    # 初始化tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # 设置预分词器
    tokenizer.pre_tokenizer = Whitespace()

    # 配置训练器
    trainer = BpeTrainer(
        vocab_size=8000,  # 词汇表大小
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        # min_frequency=2
    )

    # 训练tokenizer
    print("开始训练tokenizer...")
    tokenizer.train_from_iterator(get_training_corpus(dataset), trainer=trainer)

    # 添加后处理（用于分类任务）
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    # 创建保存目录
    os.makedirs('./ag_news_tokenizer', exist_ok=True)

    # 保存tokenizer
    tokenizer.save("./ag_news_tokenizer/tokenizer.json")
    print("tokenizer训练完成并保存!")

    
# # 测试tokenizer
# test_text = "This is a test sentence for tokenizer."
# encoding = tokenizer.encode(test_text)
# print(f"\n测试文本: {test_text}")
# print(f"Tokenized: {encoding.tokens}")
# print(f"IDs: {encoding.ids}")