#!/bin/bash
# ==========================
# 运行 MLM 预训练任务
# ==========================

# 项目根目录
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
SRC_DIR="$PROJECT_ROOT/src"
CONFIG_DIR="$PROJECT_ROOT/configs"

# 创建保存目录
mkdir -p "$PROJECT_ROOT/save/model/MLM/checkpoints"
mkdir -p "$PROJECT_ROOT/results/MLM"

# 激活虚拟环境（如果需要）
# source ~/envs/mlm_env/bin/activate

# 运行脚本
echo "🚀 启动 MLM 训练 ..."
python "$SRC_DIR/mlm_pretraining.py" --config "$CONFIG_DIR/pretrain.yaml"

echo "✅ 训练完成！结果保存在 ./save/ 和 ./results/ 下。"
