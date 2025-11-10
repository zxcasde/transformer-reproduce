from datasets import load_dataset
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os

# 准备训练数据
def get_training_corpus(dataset):
    for i in range(0, len(dataset['train']), 1000):  # 分批处理避免内存问题
        yield dataset['train'][i:i+1000]['text']

def main(datasets_name, save_name, cache_dir='./data'):
    # 加载数据集
    dataset = load_dataset(*datasets_name, cache_dir=cache_dir)

    # 初始化tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

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
    os.makedirs(f'./datasets/tokenizer', exist_ok=True)

    # 保存tokenizer
    tokenizer.save(f"./datasets/tokenizer/{save_name}_tokenizer.json")
    print("tokenizer训练完成并保存!")

if __name__=='__main__':

    main(("ag_news", None), "ag_news")
    main(("Salesforce/wikitext", "wikitext-2-raw-v1"), "wikitext")

