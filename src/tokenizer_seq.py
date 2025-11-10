from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace

from datasets import load_dataset

def get_data():
    dataset = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-de", cache_dir='./data')

    corpus_src, corpus_tgt = [], []
    for item in dataset['train']['translation']:
        corpus_src.append(item['en'])
        corpus_tgt.append(item['de'])

    return corpus_src, corpus_tgt

def train_tokenizer(file_path, vocab_size=16000, save_path="tokenizer.json"):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"]
    )
    tokenizer.train([file_path], trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<sos> $A <eos>",
        special_tokens=[("<sos>", tokenizer.token_to_id("<sos>")), ("<eos>", tokenizer.token_to_id("<eos>"))],
    )
    tokenizer.save(save_path)
    return tokenizer


if __name__=='__main__':
    corpus_src, corpus_tgt = get_data()
    
    with open("src.txt", "w") as f_src:
        f_src.write("\n".join(corpus_src))

    with open("tgt.txt", "w") as f_tgt:
        f_tgt.write('\n'.join(corpus_tgt))

    tokenizer_src = train_tokenizer("src.txt", save_path="./datasets/tokenizer/src_tokenizer.json")
    tokenizer_tgt = train_tokenizer("tgt.txt", save_path="./datasets/tokenizer/tgt_tokenizer.json")

    # print(len(loader))

    # # 示例：分词 + 编码 + 解码
    encoded = tokenizer_src.encode("I love you")
    print(encoded.tokens)
    # print(encoded.ids)
    # print(tokenizer_src.decode(encoded.ids))
