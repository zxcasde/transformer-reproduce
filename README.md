
## Getting Started
Clone repo:
``` bash
git clone git@github.com:facebookresearch/coconut.git
cd coconut
```

# Quick Start

## 1. Setup environment:
``` bash
conda create -n transformer python=3.10
conda activate transformer
```

## 2. Install the requirements
```pip install -r requirements.txt```

## 3. Run the following command:
``` bash
bash scripts/run.sh
```


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
