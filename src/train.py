import yaml
import argparse
import os 
import torch
import platform

print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def update_config(config, args):
    """用命令行参数更新配置"""
    for key, value in vars(args).items():
        if value is not None and key in config.get('training', {}):
            config['training'][key] = value
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--batch_size', type=int, help='批量大小')
    parser.add_argument('--lr', type=float, help='学习率')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 更新配置
    if args.lr:
        config['training']['optimizer']['learning_rate'] = args.lr
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    if 'device' in config and 'gpus' in config['device']:
        gpus = str(config['device']['gpus'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        print(f"[INFO] Using GPUs: {gpus}")

    # 3️⃣ 检查 CUDA 是否可用
    use_cuda = config['device'].get('use_cuda', True)
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    print(f"[INFO] Running on device: {device}")

    print("最终配置:", config)