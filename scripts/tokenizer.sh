#!/bin/bash

# 训练 encoder_only 架构上预训练和微调所需的分词器
python src/tokenizer_bert.py

# 训练 encoder_decoder 架构上源语言和目标语言所需的分词器
python src/tokenizer_seq.py