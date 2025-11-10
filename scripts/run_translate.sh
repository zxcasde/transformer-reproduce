#!/bin/bash
# ==========================================================
# 🚀 Transformer Seq2Seq: 训练 + 评估 一体化流程
# ==========================================================

# ==== 环境设置 ====
set -e  # 遇到错误立即停止
export TOKENIZERS_PARALLELISM=false

PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
SRC_DIR="$PROJECT_ROOT/src"
CONFIG_DIR="$PROJECT_ROOT/configs"
RESULTS_DIR="$PROJECT_ROOT/results/seq2seq"
MODEL_DIR="$PROJECT_ROOT/save/model/seq2seq"
EVAL_RESULTS_DIR="$RESULTS_DIR/evaluation_results"

mkdir -p "$MODEL_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$EVAL_RESULTS_DIR"

# ==== 第一步：训练模型 ====
echo "=============================================="
echo "🧠 Step 1: 开始训练 Transformer Seq2Seq 模型"
echo "=============================================="

python "$SRC_DIR/train_seq_to_seq.py" \
  --config "$CONFIG_DIR/translate.yaml"

echo "✅ 训练完成，模型已保存至: $MODEL_DIR"
echo

# ==== 第二步：评估所有模型 ====
echo "=============================================="
echo "📊 Step 2: 开始评估所有保存的模型"
echo "=============================================="

python "$SRC_DIR/eval_seq_to_seq.py" \
  --model_dir "$MODEL_DIR" \
  --max_epochs 50 \
  --max_batches 20 \
  --results_dir "$EVAL_RESULTS_DIR"

echo
echo "✅ 模型评估完成，结果保存在: $EVAL_RESULTS_DIR"
echo

# ==== 第三步：查看最佳模型 ====
BEST_INFO_FILE="$EVAL_RESULTS_DIR/best_model_info.json"

if [ -f "$BEST_INFO_FILE" ]; then
  echo "=============================================="
  echo "🏆 Step 3: 最佳模型信息"
  echo "=============================================="
  cat "$BEST_INFO_FILE"
else
  echo "⚠️ 未找到最佳模型信息文件: $BEST_INFO_FILE"
fi

echo
echo "🎉 全流程完成！模型已训练、评估、绘图并生成最佳结果。"
