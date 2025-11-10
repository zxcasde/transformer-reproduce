import torch
from torch.serialization import add_safe_globals
from transformer_h import Encoder, Encoder_Only_Transformer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from datasets import load_dataset
from tokenizers import Tokenizer
import seaborn as sns
import numpy as np
from pathlib import Path
import random
import yaml

def get_tokenizer(datasets_name):
    tokenizer_src = Tokenizer.from_file(f"./datasets/tokenizer/{datasets_name}_tokenizer.json")
    return tokenizer_src

def get_data(datasets_name, load_local=False, cache_dir='./data'):
    if load_local:
        train_texts = []
        train_labels = []
        test_texts = []
        test_labels = []
        
        train_json_path = f'./datasets/{datasets_name}_train.json'
        with open(train_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                train_texts.append(data["text"])
                train_labels.append(data.get("label", None))
        
        test_json_path = f'./datasets/{datasets_name}_test.json'
        with open(test_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                test_texts.append(data["text"])
                test_labels.append(data.get("label", None))
    else:
        dataset = load_dataset(*datasets_name, cache_dir=cache_dir)

        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']

    return (train_texts, train_labels), (test_texts, test_labels)

def create_padding_mask(seq, pad_id):
    """创建padding mask"""
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)

class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id("[PAD]")

    def pad(self, ids, pad_id):
        if len(ids) < self.max_len:
            ids = ids + [pad_id] * (self.max_len - len(ids))
        return ids[:self.max_len]

    def __getitem__(self, idx):
        # 编码文本
        text_ids = self.tokenizer.encode(self.texts[idx]).ids
        text_ids = self.pad(text_ids, self.pad_id)
        
        # 标签
        label = self.labels[idx]
        
        return torch.tensor(text_ids), torch.tensor(label)

    def __len__(self):
        return len(self.texts)

def train(model, train_loader, test_loader, tokenizer, criterion, optimizer, device, epochs, 
          model_save_dir='./models/sft', results_save_dir="./results/sft", 
          experiment_name=""):
    
    # 创建保存目录
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_save_dir, exist_ok=True)
    
    # 实验配置记录
    config = {
        'experiment_name': experiment_name,
        'start_time': datetime.now().isoformat(),
        'epochs': epochs,
        'device': str(device),
        'optimizer': optimizer.__class__.__name__,
        'criterion': criterion.__class__.__name__
    }

    # 训练历史记录 - 添加批次损失记录
    history = {
        'batch_loss': [],           # 每个批次的损失
        'epoch_train_loss': [],     # 每个epoch的平均训练损失
        'epoch_train_accuracy': [], # 每个epoch的训练准确率
        'test_loss': [],            # 每个epoch的测试损失
        'test_accuracy': [],        # 每个epoch的测试准确率
        'learning_rates': [],       # 每个epoch的学习率
        'epoch_times': [],          # 每个epoch的训练时间
        'epoch_boundaries': []      # 每个epoch开始的批次索引
    }
    
    # 最佳模型跟踪
    best_test_accuracy = 0.0
    best_test_loss = float('inf')
    best_epoch = 0
    
    print(f"开始训练实验: {experiment_name}")
    print(f"设备: {device}")
    print(f"训练样本: {len(train_loader.dataset)}, 测试样本: {len(test_loader.dataset)}")

    for epoch in range(epochs):
        epoch_start_time = datetime.now()
        
        # 记录当前epoch开始的批次索引
        current_batch_count = len(history['batch_loss'])
        history['epoch_boundaries'].append(current_batch_count)
        
        # 训练阶段
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [训练]", unit="batch")
        
        for batch_idx, (src, labels) in enumerate(progress_bar):
            src, labels = src.to(device), labels.to(device)
            
            # 创建padding mask
            src_mask = create_padding_mask(src, tokenizer.token_to_id("[PAD]")).to(device)
            
            # 前向传播
            outputs = model(src, src_mask)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 统计信息
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train_correct += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)
            
            # 记录每个batch的损失
            history['batch_loss'].append(loss.item())
            
            # 更新进度条
            avg_train_loss = total_train_loss / (batch_idx + 1)
            train_accuracy = total_train_correct / total_train_samples
            current_lr = optimizer.param_groups[0]['lr']
            
            progress_bar.set_postfix({
                "loss": f"{avg_train_loss:.4f}",
                "acc": f"{train_accuracy:.4f}",
                "lr": f"{current_lr:.2e}"
            })
        
        progress_bar.close()
        
        # 记录训练统计
        epoch_train_loss = total_train_loss / len(train_loader)
        epoch_train_accuracy = total_train_correct / total_train_samples
        history['epoch_train_loss'].append(epoch_train_loss)
        history['epoch_train_accuracy'].append(epoch_train_accuracy)
        history['learning_rates'].append(current_lr)
        
        # 测试集评估
        test_loss, test_accuracy = evaluate_model(model, test_loader, tokenizer, criterion, device)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)
        
        # 计算epoch时间
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        history['epoch_times'].append(epoch_time)
        
        # 打印epoch总结
        print(f"Epoch {epoch+1} 总结:")
        print(f"  训练损失: {epoch_train_loss:.4f}, 训练准确率: {epoch_train_accuracy:.4f}")
        print(f"  测试损失: {test_loss:.4f}, 测试准确率: {test_accuracy:.4f}")
        print(f"  学习率: {current_lr:.2e}, 时间: {epoch_time:.2f}s")
        print(f"  批次数量: {len(train_loader)}, 总批次: {len(history['batch_loss'])}")
        
        # 保存checkpoint
        checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'train_accuracy': epoch_train_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'history': history
        }, checkpoint_path)
        
        # 保存最佳模型
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_test_loss = test_loss
            best_epoch = epoch + 1
            
            best_model_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'test_loss': best_test_loss,
                'test_accuracy': best_test_accuracy,
                'config': config
            }, best_model_path)
            print(f"  新的最佳模型已保存! 测试准确率: {best_test_accuracy:.4f}")
    
    # 更新配置信息
    config['end_time'] = datetime.now().isoformat()
    config['total_training_time'] = sum(history['epoch_times'])
    config['best_epoch'] = best_epoch
    config['best_test_loss'] = best_test_loss
    config['best_test_accuracy'] = best_test_accuracy
    config['final_epoch'] = epoch + 1
    config['total_batches'] = len(history['batch_loss'])
    
    # 保存训练历史和配置
    history_path = os.path.join(results_save_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    
    config_path = os.path.join(results_save_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    # 保存训练曲线数据（便于绘图）
    curve_data = {
        'epochs': list(range(1, len(history['epoch_train_loss']) + 1)),
        'batch_loss': history['batch_loss'],
        'epoch_train_loss': history['epoch_train_loss'],
        'test_loss': history['test_loss'],
        'epoch_train_accuracy': history['epoch_train_accuracy'],
        'test_accuracy': history['test_accuracy'],
        'learning_rates': history['learning_rates'],
        'epoch_boundaries': history['epoch_boundaries']
    }
    
    curve_path = os.path.join(results_save_dir, 'training_curves.json')
    with open(curve_path, 'w', encoding='utf-8') as f:
        json.dump(curve_data, f, indent=4, ensure_ascii=False)
    
    print(f"实验 '{experiment_name}' 完成!")
    print(f"最佳模型在 epoch {best_epoch}, 测试准确率: {best_test_accuracy:.4f}")
    
    return history, config

def setup_plot_style():
    """设置绘图样式"""
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['font.family'] = 'DejaVu Sans'
    sns.set_style("whitegrid")

def plot_batch_loss(history_dict, save_path=None, show=True, smoothing=0.8):
    """
    绘制批次级别的损失下降曲线
    """
    setup_plot_style()
    
    # 如果传入的是文件路径，则加载数据
    if isinstance(history_dict, str):
        with open(history_dict, 'r', encoding='utf-8') as f:
            history_dict = json.load(f)
    
    if 'batch_loss' not in history_dict:
        print("未找到批次损失数据")
        return
    
    batch_losses = history_dict['batch_loss']
    batches = range(1, len(batch_losses) + 1)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制原始批次损失
    plt.plot(batches, batch_losses, 'b-', alpha=0.3, linewidth=1, label='Batch Loss')
    
    # 计算滑动平均（平滑曲线）
    if smoothing > 0:
        smoothed_losses = []
        last_smooth = batch_losses[0]
        for loss in batch_losses:
            smoothed = last_smooth * smoothing + loss * (1 - smoothing)
            smoothed_losses.append(smoothed)
            last_smooth = smoothed
        
        plt.plot(batches, smoothed_losses, 'r-', linewidth=2, 
                label=f'Smoothed Loss (α={smoothing})')
    
    # 标记epoch边界
    if 'epoch_boundaries' in history_dict:
        for i, boundary in enumerate(history_dict['epoch_boundaries']):
            if boundary < len(batch_losses):
                plt.axvline(x=boundary, color='g', linestyle='--', alpha=0.5)
                if i == 0:  # 只在第一个边界添加标签
                    plt.text(boundary, np.max(batch_losses) * 0.9, 
                            'Epoch Start', rotation=90, ha='right', color='g')
    
    plt.title('Training Loss per Batch')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 设置x轴格式，避免显示过多刻度
    max_batches = len(batch_losses)
    if max_batches > 1000:
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Batch loss plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_batch_loss_comparison(experiments_data, save_path=None, show=True, smoothing=0.9):
    """
    比较多个实验的批次损失
    """
    setup_plot_style()
    
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_data)))
    
    for (exp_name, history), color in zip(experiments_data.items(), colors):
        # 如果传入的是文件路径，则加载数据
        if isinstance(history, str):
            with open(history, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        if 'batch_loss' not in history:
            continue
            
        batch_losses = history['batch_loss']
        batches = range(1, len(batch_losses) + 1)
        
        # 计算滑动平均
        if smoothing > 0:
            smoothed_losses = []
            last_smooth = batch_losses[0]
            for loss in batch_losses:
                smoothed = last_smooth * smoothing + loss * (1 - smoothing)
                smoothed_losses.append(smoothed)
                last_smooth = smoothed
            
            plt.plot(batches, smoothed_losses, color=color, linewidth=2, label=exp_name)
        else:
            plt.plot(batches, batch_losses, color=color, alpha=0.7, linewidth=1, label=exp_name)
    
    plt.title('Batch Loss Comparison Across Experiments')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Batch loss comparison saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_epoch_metrics(history_dict, save_path=None, show=True):
    """
    绘制epoch级别的指标（损失和准确率）
    """
    setup_plot_style()
    
    # 如果传入的是文件路径，则加载数据
    if isinstance(history_dict, str):
        with open(history_dict, 'r', encoding='utf-8') as f:
            history_dict = json.load(f)
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history_dict['epoch_train_loss']) + 1)
    
    # 绘制损失曲线
    ax1.plot(epochs, history_dict['epoch_train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o')
    if 'test_loss' in history_dict:
        ax1.plot(epochs, history_dict['test_loss'], 'r-', label='Test Loss', linewidth=2, marker='s')
    
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    ax2.plot(epochs, history_dict['epoch_train_accuracy'], 'b-', label='Training Accuracy', linewidth=2, marker='o')
    if 'test_accuracy' in history_dict:
        ax2.plot(epochs, history_dict['test_accuracy'], 'r-', label='Test Accuracy', linewidth=2, marker='s')
    
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Epoch metrics plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_sample_predictions(model, data_loader, tokenizer, device, num_samples=3, save_path=None, show=False):
    """绘制样本预测结果"""
    setup_plot_style()

    model.eval()
    samples_shown = 0
    
    # 创建图形
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    with torch.no_grad():
        for batch_idx, (src, labels) in enumerate(data_loader):
            if samples_shown >= num_samples:
                break
                
            src, labels = src.to(device), labels.to(device)
            src_mask = create_padding_mask(src, tokenizer.token_to_id("[PAD]")).to(device)
            
            outputs = model(src, src_mask)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(min(src.size(0), num_samples - samples_shown)):
                ax = axes[samples_shown]
                
                input_text = tokenizer.decode(src[i].cpu().numpy())
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                confidence = torch.softmax(outputs[i], 0)[pred_label].item()
                
                ax.text(0.1, 0.7, f"input: {input_text[:100]}...", fontsize=10)
                ax.text(0.1, 0.5, f"label: {true_label}", fontsize=10, 
                       color='green' if true_label == pred_label else 'red')
                ax.text(0.1, 0.3, f"predict: {pred_label}", fontsize=10,
                       color='green' if true_label == pred_label else 'red')
                ax.text(0.1, 0.1, f"confidence: {confidence:.4f}", fontsize=10)
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title(f"sample {samples_shown + 1}", fontsize=12)
                
                samples_shown += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"样本预测图已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def evaluate_model(model, data_loader, tokenizer, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for src, labels in tqdm(data_loader, desc="评估"):
            src, labels = src.to(device), labels.to(device)
            
            src_mask = create_padding_mask(src, tokenizer.token_to_id("[PAD]")).to(device)
            
            outputs = model(src, src_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"✅ 配置文件已加载: {config_path}")
    return config

# -------- 固定随机种子 --------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed fixed to {seed}]")

# ===============================================================
# 主函数 (支持外部配置)
# ===============================================================
def main_with_config(config_path="./configs/sft_config.yaml", load_local=False):
    # 加载配置
    config = load_config(config_path)

    # 基本信息
    dataset_name = config["dataset_name"]
    device = torch.device(config["device"])
    batch_size = config["batch_size"]
    test_batch_size = config.get("test_batch_size", 32)
    epochs = config["epochs"]
    d_model = config["d_model"]
    n_heads = config["n_heads"]
    d_ff = config["d_ff"]
    num_layers = config["num_layers"]
    learning_rate = config["learning_rate"]
    pretrained_path = config["pretrained_path"]
    num_classes = config["num_classes"]
    experiment_name = config.get("experiment_name", "sft_experiment")

    # 加载 tokenizer 和数据
    tokenizer = get_tokenizer(dataset_name)
    (train_texts, train_labels), (test_texts, test_labels) = get_data((dataset_name, None), load_local)

    # 数据集 & DataLoader
    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer)
    test_dataset = ClassificationDataset(test_texts, test_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    vocab_size = tokenizer.get_vocab_size()
    print(f"🧠 Vocab size: {vocab_size}, Num classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    add_safe_globals([Tokenizer])

    # 加载预训练 encoder
    print(f"🔹 加载预训练权重: {pretrained_path}")
    ckpt = torch.load(pretrained_path, weights_only=False, map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    encoder = Encoder(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        d_ff=d_ff, num_layers=num_layers, dropout=0.1
    )
    missing, unexpected = encoder.load_state_dict(
        {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")},
        strict=False
    )
    print("Loaded pretrained encoder (missing/unexpected):", missing, unexpected)

    # 分类模型
    model = Encoder_Only_Transformer(encoder, d_model=d_model, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"💡 Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🚀 开始训练 {experiment_name}")

    history, config_log = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        experiment_name=experiment_name,
        model_save_dir="./save/model/sft",
        results_save_dir="./results/sft"
    )

    # 绘制结果
    results_dir = "./results/sft"
    plot_batch_loss(
        history_dict=os.path.join(results_dir, 'training_curves.json'),
        save_path=os.path.join(results_dir, f'{experiment_name}_batch_loss.png'),
        smoothing=0.95
    )
    plot_epoch_metrics(
        history_dict=os.path.join(results_dir, 'training_curves.json'),
        save_path=os.path.join(results_dir, f'{experiment_name}_epoch_metrics.png')
    )
    plot_sample_predictions(
        model=model,
        data_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        num_samples=3,
        save_path=os.path.join(results_dir, f'{experiment_name}_sample_predictions.png')
    )

    print("✅ Fine-tuning completed!")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Transformer classifier with pre-trained encoder")
    parser.add_argument("--config", type=str, default="./config/sft_config.yaml", help="Path to YAML config file")
    parser.add_argument("--load_local", action="store_true", help="Load local dataset instead of HuggingFace")
    args = parser.parse_args()

    set_seed()
    main_with_config(config_path=args.config, load_local=args.load_local)