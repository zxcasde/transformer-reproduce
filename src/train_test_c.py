import torch
from transformer_h import Encoder_Only_Transformer
from tokenizer_classify import get_tokenizer, get_data
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

def create_padding_mask(seq, pad_id):
    """创建padding mask"""
    return (seq == pad_id).unsqueeze(1).unsqueeze(2)

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

def train(model, loader, criterion, optimizer, device, epochs, model_save_dir='../models/ag_news', results_save_dir="../results/ag_news"):
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_save_dir, exist_ok=True)

    history = {
        'train_loss': [],
        'train_accuracy': [],
    }
    
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        
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
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = total_correct / total_samples
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{accuracy:.4f}"
            })
            
            # 记录每个batch的损失
            history['train_loss'].append(loss.item())
        
        # 记录每个epoch的准确率
        epoch_accuracy = total_correct / total_samples
        history['train_accuracy'].append(epoch_accuracy)
        
        progress_bar.close()
        
        # 保存checkpoint
        checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'accuracy': epoch_accuracy,
        }, checkpoint_path)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={epoch_accuracy:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
    
    # 保存训练历史
    history_path = os.path.join(results_save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    return history

def plot_training_history(history, results_save_dir="../results/ag_news"):
    """绘制训练损失和准确率曲线"""
    os.makedirs(results_save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线（按batch）
    batches = range(len(history['train_loss']))
    ax1.plot(batches, history['train_loss'], label='Training Loss', linewidth=1, color='blue', alpha=0.7)
    ax1.set_title('Training Loss Over Batches', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 如果损失值范围很大，使用对数刻度
    if len(history['train_loss']) > 0 and max(history['train_loss']) / (min(history['train_loss']) + 1e-8) > 100:
        ax1.set_yscale('log')
        ax1.set_ylabel('Loss (log scale)')
    
    # 绘制准确率曲线（按epoch）
    epochs = range(len(history['train_accuracy']))
    ax2.plot(epochs, history['train_accuracy'], label='Training Accuracy', linewidth=2, color='green', marker='o')
    ax2.set_title('Training Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)  # 准确率在0-1之间
    
    plt.tight_layout()
    
    # 保存图像
    plot_filename = os.path.join(results_save_dir, f'training_metrics_{timestamp}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {plot_filename}")

if __name__ == '__main__':
    # 加载tokenizer和数据
    tokenizer = get_tokenizer()
    (train_texts, train_labels), (test_texts, test_labels) = get_data()
    
    # 创建数据集
    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer)
    # 如果需要验证集，可以使用测试集
    # val_dataset = ClassificationDataset(test_texts, test_labels, tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Number of classes: 4")  # AG News有4个类别
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化模型
    model = Encoder_Only_Transformer(
        src_vocab=tokenizer.get_vocab_size(),
        num_classes=4,  # AG News的类别数
        d_model=512,
        n_heads=8,
        d_ff=3072,
        num_layers=6,
        dropout=0.1
    ).to(device)
    
    # 损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # 学习率调度器（可选）
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    history = train(model, train_loader, criterion, optimizer, device, epochs=3)
    
    # 绘制训练曲线
    plot_training_history(history)
    
    print("Training completed!")