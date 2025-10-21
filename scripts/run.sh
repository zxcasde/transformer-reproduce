#!/bin/bash
# 退出时如果有错误则终止
set -e

# 定位到项目根目录（run.sh 在 scripts/ 里）
cd "$(dirname "$0")/.."

# 运行 train.py，并传入配置文件路径
python src/train.py --config configs/base.yaml
