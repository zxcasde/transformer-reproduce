
# 快速开始
Clone repo:
``` bash
git clone git@github.com:zxcasde/transformer-reproduce.git
cd transformer-reproduce
```

# 启动

## 1. 环境准备:
``` bash
conda create -n transformer python=3.10
conda activate transformer
```

## 2. 安装依赖
```pip install -r requirements.txt```

## 3. Huggingface 准备数据，或直接从本地加载
数据集原始链接：
https://huggingface.co/datasets/Salesforce/wikitext

https://huggingface.co/datasets/fancyzhx/ag_news

https://huggingface.co/datasets/IWSLT/iwslt2017

提供两种方式，huggingface加载速度比本地加载更快，默认huggingface加载，请运行以下下载脚本

```python download.py```

## 4. 训练分词器:
``` 
bash scripts/tokenizer.sh
```


## 5.

## 6.

# 环境与硬件配置
本项目基于 PyTorch 复现 Transformer 模型，推荐硬件与环境如下：

| 项目 | 推荐配置 |
|------|-----------|
| 操作系统 | Ubuntu 20.04 |
| Python | 3.10 及以上 |
| PyTorch | 2.9.0 (支持 CUDA 12.8) |
| CUDA Toolkit | 12.8 |
| cuDNN | 9.10.2.21 |
| GPU | NVIDIA RTX 3090 |
| CPU | ≥ 8 核 |
| 内存 | ≥ 32GB |
| 磁盘空间 | ≥ 20GB |
