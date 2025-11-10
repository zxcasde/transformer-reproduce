import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import os
import json
import yaml 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random

def get_tokenizer(datasets_name):
    tokenizer_src = Tokenizer.from_file(f"./datasets/tokenizer/{datasets_name}_tokenizer.json")
    return tokenizer_src

class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer: Tokenizer, max_length=128, mlm_prob=0.15):
        self.texts = [t for t in texts if t and t.strip()]
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.mlm_prob = mlm_prob

        self.pad_id = self.tokenizer.token_to_id('[PAD]')
        self.mask_id = self.tokenizer.token_to_id('[MASK]')
        self.unk_id = self.tokenizer.token_to_id('[UNK]')
        self.cls_id = self.tokenizer.token_to_id('[CLS]')
        self.sep_id = self.tokenizer.token_to_id('[SEP]')

    def __len__(self):
        return len(self.texts)
    
    def encode_text(self, text):
        encode_ids = self.tokenizer.encode(text)
        input_ids = encode_ids.ids

        input_ids = input_ids[: self.max_len]
        if len(input_ids) < self.max_len:
            input_ids = input_ids + [self.pad_id] * (self.max_len - len(input_ids))
        attention_mask = [1 if ids != self.pad_id else 0 for ids in input_ids]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)
    
    def mask_tokens(self, input_ids):
        device = input_ids.device
        labels = input_ids.clone()

        # 随机选出15%的token用于mask
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        special_tokens_mask = torch.tensor(
            [1 if id in [self.cls_id, self.sep_id, self.pad_id] else 0 for id in labels],
            dtype=torch.bool, device=device
        )
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 非mask token不计算loss
        labels[~masked_indices] = -100

        # 80% [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_id

        # 10% 随机token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1, device=device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(0, self.tokenizer.get_vocab_size(), labels.shape, dtype=torch.long, device=device)
        input_ids[indices_random] = random_words[indices_random]

        # 10% 保留原样
        return input_ids, labels

    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids, attention_mask = self.encode_text(text)
        input_ids, labels = self.mask_tokens(input_ids.clone())

        return {'input_ids':input_ids, 'attention_mask': attention_mask, 'labels':labels}

class MLMEvaluator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    @torch.no_grad()
    def evaluate(self, dataloader, max_batches=None):
        """评估MLM模型的各项指标"""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        masked_tokens = 0
        
        progress_bar = tqdm(dataloader, desc="Evaluating MLM", unit="batch")
        
        for batch_idx, batch in enumerate(progress_bar):
            if max_batches and batch_idx >= max_batches:
                break
                
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device).unsqueeze(1).unsqueeze(2)
            labels = batch["labels"].to(self.device)
            
            # 前向传播
            logits = self.model(input_ids, attention_mask)
            
            # 计算损失
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            
            # 计算准确率（只考虑被mask的token）
            masked_positions = labels != -100
            masked_tokens_batch = masked_positions.sum().item()
            masked_tokens += masked_tokens_batch
            
            if masked_tokens_batch > 0:
                predictions = logits.argmax(dim=-1)
                correct_batch = (predictions[masked_positions] == labels[masked_positions]).sum().item()
                correct_predictions += correct_batch
            
            total_tokens += labels.numel()
            
            # 更新进度条
            current_loss = total_loss / (batch_idx + 1)
            current_acc = correct_predictions / masked_tokens if masked_tokens > 0 else 0
            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "acc": f"{current_acc:.4f}"
            })
        
        # 计算最终指标
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / masked_tokens if masked_tokens > 0 else 0
        perplexity = math.exp(avg_loss)
        
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'masked_tokens_count': masked_tokens,
            'total_tokens_count': total_tokens,
            'mask_ratio': masked_tokens / total_tokens if total_tokens > 0 else 0
        }
        
        return metrics

class TransformerLRScheduler:
    """Transformer论文中的学习率调度器：线性预热 + 逆平方根衰减"""
    
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        
    def step(self):
        """更新学习率"""
        self._step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def _get_lr(self):
        """计算当前学习率"""
        step = self._step
        warmup = self.warmup_steps
        
        # Transformer论文中的学习率公式
        lr = self.factor * (self.d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)
        return lr
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [self._get_lr()]

def create_experiment_dirs():
    """创建实验目录结构 - 简化版本，config、log、plot合并到results/MLM"""
    # 模型保存目录
    model_dirs = {
        'models': './save/model/MLM',
        'checkpoints': './save/model/MLM/checkpoints'
    }
    
    result_dirs = {
        'results': './results/MLM'
    }
    
    # 合并所有目录
    dirs = {**model_dirs, **result_dirs}
    
    # 创建所有目录
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    return dirs

def save_training_config(config, results_dir):
    """保存训练配置到results目录"""
    config_path = os.path.join(results_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"训练配置已保存至: {config_path}")

def plot_mlm_training_metrics(training_history, save_dir, experiment_name):
    """绘制MLM训练指标并保存到results目录"""
    # 设置样式
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 12
    sns.set_style("whitegrid")
    
    epochs = list(range(1, len(training_history['train_loss']) + 1))
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 绘制损失
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o')
    if 'val_loss' in training_history:
        ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制困惑度
    train_ppl = [math.exp(loss) for loss in training_history['train_loss']]
    ax2.plot(epochs, train_ppl, 'b-', label='Training PPL', linewidth=2, marker='o')
    if 'val_loss' in training_history:
        val_ppl = [math.exp(loss) for loss in training_history['val_loss']]
        ax2.plot(epochs, val_ppl, 'r-', label='Validation PPL', linewidth=2, marker='s')
    ax2.set_title('Perplexity (PPL)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 绘制准确率
    if 'train_accuracy' in training_history:
        ax3.plot(epochs, training_history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2, marker='o')
    if 'val_accuracy' in training_history:
        ax3.plot(epochs, training_history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2, marker='s')
    ax3.set_title('Masked Token Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 绘制学习率
    if 'learning_rates' in training_history:
        ax4.plot(epochs, training_history['learning_rates'], 'g-', label='Learning Rate', linewidth=2, marker='o')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'MLM Training Metrics - {experiment_name}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片到results目录
    plot_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"训练指标图已保存至: {plot_path}")
    
    plt.show()

def train_mlm_with_scheduler(model, train_loader, val_loader, tokenizer, optimizer, 
                           scheduler, criterion, device, epochs, save_dirs, experiment_name):
    """训练MLM模型，使用学习率调度器和完整保存功能"""
    
    training_history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'train_perplexity': [],
        'val_perplexity': [],
        'learning_rates': []
    }
    
    evaluator = MLMEvaluator(model, tokenizer, device)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_masked = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).unsqueeze(1).unsqueeze(2)
            labels = batch["labels"].to(device)
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 更新学习率
            if scheduler:
                current_lr = scheduler.step()
            
            # 统计训练准确率
            masked_positions = labels != -100
            if masked_positions.sum() > 0:
                predictions = logits.argmax(dim=-1)
                train_correct += (predictions[masked_positions] == labels[masked_positions]).sum().item()
                train_masked += masked_positions.sum().item()
            
            total_train_loss += loss.item()
            
            # 更新进度条
            avg_loss = total_train_loss / (batch_idx + 1)
            train_acc = train_correct / train_masked if train_masked > 0 else 0
            current_lr = optimizer.param_groups[0]['lr'] if not scheduler else current_lr
            
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{train_acc:.4f}",
                "lr": f"{current_lr:.2e}"
            })
        
        # 记录训练指标
        epoch_train_loss = total_train_loss / len(train_loader)
        epoch_train_acc = train_correct / train_masked if train_masked > 0 else 0
        
        training_history['train_loss'].append(epoch_train_loss)
        training_history['train_accuracy'].append(epoch_train_acc)
        training_history['train_perplexity'].append(math.exp(epoch_train_loss))
        training_history['learning_rates'].append(current_lr)
        
        # 验证阶段
        if val_loader:
            val_metrics = evaluator.evaluate(val_loader)
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
            training_history['val_perplexity'].append(val_metrics['perplexity'])
            
            print(f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f}")
            print(f"          Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}")
            
            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_path = os.path.join(save_dirs['models'], 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler._step if scheduler else None,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'training_history': training_history
                }, best_model_path)
                print(f"  新的最佳模型已保存! 验证损失: {best_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f}")
        
        # 保存检查点
        checkpoint_path = os.path.join(save_dirs['checkpoints'], f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler._step if scheduler else None,
            'training_history': training_history
        }, checkpoint_path)
    
    # 保存最终模型
    final_model_path = os.path.join(save_dirs['models'], 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': training_history,
        'tokenizer': tokenizer
    }, final_model_path)
    
    # 保存训练历史到results目录
    history_path = os.path.join(save_dirs['results'], 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=4, ensure_ascii=False)
    
    print(f"训练完成! 所有文件已保存至指定目录")
    return training_history

# ============================ 加载配置部分 ============================
def load_config(config_path: str):
    """从 YAML 文件加载训练配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"配置文件已加载: {config_path}")
    return config

# ============================ 主训练函数修改 ============================
def main_with_scheduler(config_path='./config/config.yaml', load_local=False):
    from transformer_h import Encoder_Only_MLM
    
    # 读取配置
    config = load_config(config_path)
    experiment_name = config.get('experiment_name', 'mlm_pretraining')
    
    # 创建目录
    save_dirs = create_experiment_dirs()
    save_training_config(config, save_dirs['results'])
    
    # 加载数据
    tokenizer = get_tokenizer(config['dataset_name'])
    print(f"Loading {config['dataset_name']} dataset...")

    if load_local:
        train_texts, val_texts = [], []
        with open('./datasets/wikitext_train.json', 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                train_texts.append(data["text"])
        with open('./datasets/wikitext_validation.json', 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                val_texts.append(data["text"])
    else:
        datasets = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", cache_dir='./data')
        train_texts = datasets["train"]["text"]
        val_texts = datasets["validation"]["text"]
    
    # 创建 Dataset / DataLoader
    train_dataset = MLMDataset(train_texts, tokenizer, 
                              max_length=config['max_length'], mlm_prob=config['mlm_prob'])
    val_dataset = MLMDataset(val_texts, tokenizer, 
                            max_length=config['max_length'], mlm_prob=config['mlm_prob'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    vocab_size = tokenizer.get_vocab_size()
    device = torch.device(config['device'])
    print("Vocab size:", vocab_size, "device:", device)
    
    # 创建模型
    model = Encoder_Only_MLM(
        src_vocab=vocab_size, 
        d_model=config['d_model'], 
        n_heads=config['n_heads'], 
        d_ff=config['d_ff'], 
        num_layers=config['num_layers']
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(optimizer, d_model=config['d_model'], warmup_steps=config['warmup_steps'])
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    print("开始训练MLM模型...")
    training_history = train_mlm_with_scheduler(
        model, train_loader, val_loader, tokenizer, optimizer, scheduler, 
        criterion, device, config['epochs'], save_dirs, experiment_name
    )
    
    plot_mlm_training_metrics(training_history, save_dirs['results'], experiment_name)
    return training_history, save_dirs

# ============================ 固定随机种子 ============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[Seed fixed to {seed}]")

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLM Pretraining with Transformer Scheduler")
    parser.add_argument("--config", type=str, default="./config/config.yaml",
                        help="Path to YAML config file (default: ./config/config.yaml)")
    parser.add_argument("--load_local", action="store_true",
                        help="Load local dataset JSON instead of HuggingFace dataset")
    args = parser.parse_args()

    # 固定随机种子
    set_seed()

    # 运行训练
    history, dirs = main_with_scheduler(config_path=args.config, load_local=args.load_local)