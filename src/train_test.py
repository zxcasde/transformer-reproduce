import torch
from transformer_h import Encoder_Decoder_Transformer
from tokenizer_seq2seq import get_tokenizer_data
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

def create_padding_mask(seq, pad_id):
    return (seq == pad_id).unsqueeze(1).unsqueeze(2)

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer_src, tokenizer_tgt, max_len=128):
        self.src = src_texts
        self.tgt = tgt_texts
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_len = max_len
        self.pad_id_src = tokenizer_src.token_to_id("<pad>")
        self.pad_id_tgt = tokenizer_tgt.token_to_id("<pad>")

    def pad(self, ids, pad_id):
        if len(ids) < self.max_len:
            ids += [pad_id] * (self.max_len - len(ids))
        return ids[:self.max_len]

    def __getitem__(self, idx):
        src_ids = self.tokenizer_src.encode(self.src[idx]).ids
        tgt_ids = self.tokenizer_tgt.encode(self.tgt[idx]).ids
        src_ids = self.pad(src_ids, self.pad_id_src)
        tgt_ids = self.pad(tgt_ids, self.pad_id_tgt)
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

    def __len__(self):
        return len(self.src)

def train(model, loader, criterion, optimizer, device, epochs, model_save_dir='../models', results_save_dir="../results"):
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_save_dir, exist_ok=True)

    history = {
        'train_loss': [],
        # 'val_loss': [],
    }
    
    best_val_loss = float('inf')
    best_model_path = ""

    for epoch in range(epochs):
        progress_bar = tqdm(loader, desc="Training", unit="batch", 
                    postfix={"loss": "?.???"})
        model.train()
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            src_mask = create_padding_mask(src, tokenizer_src.token_to_id("<pad>")).to(device)
            memory_mask = None

            preds = model(src, tgt_input, src_mask, tgt_mask, memory_mask)
            loss = criterion(preds.reshape(-1, preds.size(-1)), tgt_output.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            # print(f"Grad norm: {torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history['train_loss'].append(loss.item())
            # 实时更新进度条信息
            progress_bar.set_postfix({
                # "avg_loss": f"{avg_loss:.4f}",
                "current_loss": f"{loss.item():.4f}"
            })
            if torch.isnan(loss):
                print("Loss is NaN, stopping.")
                break
        
        progress_bar.close()
        checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'train_loss': avg_train_loss,
        }, checkpoint_path)

        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
    
    # ===== 保存训练历史 =====
    history_path = os.path.join(results_save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    return history

def plot_training_history(history, results_save_dir="../results"):
    """
    绘制训练损失曲线（以batch为单位）并保存
    """
    os.makedirs(results_save_dir, exist_ok=True)
    
    # 创建时间戳用于文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    
    # 绘制训练损失
    batches = range(len(history['train_loss']))
    plt.plot(batches, history['train_loss'], label='Training Loss', linewidth=1.5, color='blue')
    
    # 图表标题和标签
    plt.title('Training Loss Over Batches', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 如果损失值范围很大，使用对数刻度
    if max(history['train_loss']) / min(history['train_loss']) > 100:
        plt.yscale('log')
        plt.ylabel('Loss (log scale)', fontsize=12)
    
    # 添加一些美化
    plt.tight_layout()
    
    # 保存图像
    plot_filename = os.path.join(results_save_dir, f'training_loss_batches_{timestamp}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training loss curve (by batch) saved to: {plot_filename}")
    # return plot_filename
    
if __name__== '__main__':
    tokenizer_src, tokenizer_tgt, corpus_src, corpus_tgt = get_tokenizer_data()
    dataset = TranslationDataset(corpus_src, corpus_tgt, tokenizer_src, tokenizer_tgt)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    device = "cuda:0"
    model = Encoder_Decoder_Transformer(
        src_vocab=tokenizer_src.get_vocab_size(),
        tgt_vocab=tokenizer_tgt.get_vocab_size(),
        d_model=512,
        n_heads=8,
        d_ff=3072,
        num_layers=6
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("<pad>"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # print(model)
    history = train(model, loader, criterion, optimizer, device, 2)

    plot_training_history(history)