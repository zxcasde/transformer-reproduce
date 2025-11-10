#!/bin/bash
# ==========================================================
# 运行下游分类任务 (SFT Fine-tuning)
# ==========================================================

PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
SRC_DIR="$PROJECT_ROOT/src"
CONFIG_DIR="$PROJECT_ROOT/configs"

mkdir -p "$PROJECT_ROOT/save/model/sft"
mkdir -p "$PROJECT_ROOT/results/sft"

echo "🚀 启动 SFT 训练 ..."
python "$SRC_DIR/train_sft.py" --config "$CONFIG_DIR/finetune.yaml"

echo "✅ SFT 训练完成，结果保存在 ./save/model/sft 与 ./results/sft 下"
