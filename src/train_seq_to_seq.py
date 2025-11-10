import torch
from transformer_h import Encoder_Decoder_Transformer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import random
import numpy as np
import seaborn as sns
import yaml
import argparse

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # 确保 cudnn 使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 关闭部分非确定性算子
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"[Seed fixed to {seed}]")

def transformer_lr(step, d_model, warmup=4000, eps=1e-8):
    step = max(step, eps)  # 确保step不为0
    arg1 = step ** (-0.5)
    arg2 = step * (warmup ** -1.5)
    lr = (d_model ** -0.5) * min(arg1, arg2)
    # if step % 50 == 0:
    #     print(f"Step {step}: lr = {lr:.2e}, arg1 = {arg1:.2e}, arg2 = {arg2:.2e}")
    
    return lr

def get_param_groups(model, weight_decay=0.01):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name.lower() or "emb" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def load_iwslt_from_json(dir_path):
    corpus_src_train, corpus_tgt_train = [], []
    corpus_src_test, corpus_tgt_test = [], []
    
    with open(f'{dir_path}/iwslt_train.json', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            corpus_src_train.append(data["en"])
            corpus_tgt_train.append(data["de"])
    
    with open(f'{dir_path}/iwslt_test.json', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            corpus_src_test.append(data["en"])
            corpus_tgt_test.append(data["de"])
    
    return corpus_src_train, corpus_tgt_train, corpus_src_test, corpus_tgt_test

def get_tokenizer_data(dir_path='./datasets', load_with_local_json=False):
    tokenizer_src = Tokenizer.from_file(f"{dir_path}/tokenizer/src_tokenizer.json")
    tokenizer_tgt = Tokenizer.from_file(f"{dir_path}/tokenizer/tgt_tokenizer.json")

    if load_with_local_json:
        corpus_src_train, corpus_tgt_train, corpus_src_test, corpus_tgt_test = load_iwslt_from_json(dir_path)
    else:
        dataset = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-de", cache_dir='./data')

        corpus_src_train, corpus_tgt_train = [], []
        for item in dataset['train']['translation']:
            corpus_src_train.append(item['en'])
            corpus_tgt_train.append(item['de'])

        corpus_src_test, corpus_tgt_test = [], []
        for item in dataset['test']['translation']:
            corpus_src_test.append(item['en'])
            corpus_tgt_test.append(item['de'])

    return tokenizer_src, tokenizer_tgt, corpus_src_train, corpus_tgt_train, corpus_src_test, corpus_tgt_test

def generate_square_subsequent_mask(sz):
    return torch.tril(torch.ones(sz, sz)).bool()

def create_padding_mask(seq, pad_id):
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)

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

@torch.no_grad()
def greedy_decode_batch(model, src, src_mask, max_len, tokenizer_tgt, device):
    model.eval()
    batch_size = src.size(0)
    bos_id = tokenizer_tgt.token_to_id("<sos>")
    eos_id = tokenizer_tgt.token_to_id("<eos>")

    ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
        out = model(src, ys, src_mask, tgt_mask)
        next_token = out[:, -1, :].argmax(dim=-1).unsqueeze(1)
        ys = torch.cat([ys, next_token], dim=1)
        finished |= next_token.squeeze(1).eq(eos_id)
        if finished.all():
            break

    ys[~finished, -1] = eos_id
    return ys

@torch.no_grad()
def evaluate_bleu(model, dataloader, tokenizer_src, tokenizer_tgt, device, max_len=128, max_batches=None):
    refs, hyps = [], []
    model.eval()

    pad_id = tokenizer_src.token_to_id("<pad>")
    smooth_fn = SmoothingFunction().method4

    for idx, (src, tgt) in enumerate(tqdm(dataloader, desc="Evaluating (BLEU)")):
        if (max_batches is not None) and (idx >= max_batches):
            break

        src, tgt = src.to(device), tgt.to(device)
        src_mask = create_padding_mask(src, pad_id).to(device)

        pred_batch = greedy_decode_batch(model, src, src_mask, max_len, tokenizer_tgt, device)

        for i in range(src.size(0)):
            pred_text = tokenizer_tgt.decode([int(x) for x in pred_batch[i].tolist()], skip_special_tokens=True)
            tgt_text  = tokenizer_tgt.decode([int(x) for x in tgt[i].tolist()],        skip_special_tokens=True)

            pred_text = pred_text.replace("<sos>", "").replace("<eos>", "").replace("<pad>", "").strip()
            tgt_text  = tgt_text.replace("<sos>", "").replace("<eos>", "").replace("<pad>", "").strip()

            if len(pred_text) == 0 or len(tgt_text) == 0:
                continue

            refs.append([tgt_text.split()])   # list of references (here single ref)
            hyps.append(pred_text.split())

    if len(hyps) == 0:
        print("No hypotheses collected for BLEU evaluation.")
        return 0.0

    bleu = corpus_bleu(refs, hyps, smoothing_function=smooth_fn)
    print(f"\nBLEU Score: {bleu * 100:.2f}")
    return bleu

@torch.no_grad()
def generate_text(model, src_text, tokenizer_src, tokenizer_tgt, device, max_len=128, print_steps=True):
    model.eval()
    
    pad_id = tokenizer_src.token_to_id("<pad>")
    bos_id = tokenizer_tgt.token_to_id("<sos>")
    eos_id = tokenizer_tgt.token_to_id("<eos>")
    
    src_ids = tokenizer_src.encode(src_text).ids
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = create_padding_mask(src_tensor, pad_id).to(device)
    
    ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    
    for i in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
        out = model(src_tensor, ys, src_mask, tgt_mask)
        next_token = out[:, -1, :].argmax(dim=-1).unsqueeze(1)
        ys = torch.cat([ys, next_token], dim=1)
        
        token_str = tokenizer_tgt.id_to_token(int(next_token))
        # if print_steps:
        #     print(f"Step {i+1}: {token_str}")
        
        if next_token.item() == eos_id:
            break
    
    output_ids = ys.squeeze(0).tolist()
    output_text = tokenizer_tgt.decode(output_ids, skip_special_tokens=True)
    return output_text.strip()

# 评估 val loss（不需要 BLEU，先看 loss）
@torch.no_grad()
def eval_loss(model, dataloader, criterion, tokenizer_src, device, max_batches=50):
    model.eval()
    pad_id = tokenizer_src.token_to_id("<pad>")
    total_loss = 0.0
    count = 0
    for i, (src, tgt) in enumerate(dataloader):
        if i >= max_batches: break
        src, tgt = src.to(device), tgt.to(device)
        src_mask = create_padding_mask(src, pad_id).to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        preds = model(src, tgt_input, src_mask, tgt_mask)  # (B, T, V)
        loss = criterion(preds.reshape(-1, preds.size(-1)), tgt_output.reshape(-1))
        total_loss += loss.item() * tgt_output.numel()
        count += tgt_output.numel()
    return total_loss / count

def train_model(model, trainloader, testloader, tokenizer_src, tokenizer_tgt, criterion, optimizer, scheduler, device, epochs, model_save_dir='./save/model', results_save_dir="./results/seq_to_seq"):
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_save_dir, exist_ok=True)

    history = {
        'batch_loss': [],
        'lr': []
    }

    for epoch in range(epochs):
        progress_bar = tqdm(trainloader, desc="Training", unit="batch")
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
            scheduler.step()
            history['batch_loss'].append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
        
            progress_bar.set_postfix({
                "current_loss": f"{loss.item():.4f}",
                "current_lr": f"{current_lr:.2e}"
            })
        
        progress_bar.close()
        checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

        
        
    # ===== 保存训练历史 =====
    history_path = os.path.join(results_save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    return history

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

def plot_batch_loss(history_dict, save_path=None, show=False, smoothing=0.8):
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


def plot_learning_rate_schedule(history_dict, save_path=None, show=False):
    """
    绘制学习率随批次变化的曲线
    """
    setup_plot_style()
    
    # 如果传入的是文件路径，则加载数据
    if isinstance(history_dict, str):
        with open(history_dict, 'r', encoding='utf-8') as f:
            history_dict = json.load(f)
    
    if 'lr' not in history_dict or not history_dict['lr']:
        print("未找到批次学习率数据")
        return
    
    batch_lrs = history_dict['lr']
    batches = range(1, len(batch_lrs) + 1)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制学习率曲线
    plt.plot(batches, batch_lrs, 'purple', linewidth=2, label='Learning Rate')
    
    # 标记epoch边界
    if 'epoch_boundaries' in history_dict:
        for i, boundary in enumerate(history_dict['epoch_boundaries']):
            if boundary < len(batch_lrs):
                plt.axvline(x=boundary, color='g', linestyle='--', alpha=0.5)
                if i == 0:  # 只在第一个边界添加标签
                    plt.text(boundary, np.max(batch_lrs) * 0.9, 
                            'Epoch Start', rotation=90, ha='right', color='g')
    
    plt.title('Learning Rate Schedule per Batch')
    plt.xlabel('Batch Number')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 使用对数刻度（如果学习率变化范围很大）
    if np.max(batch_lrs) / (np.min(batch_lrs) + 1e-8) > 100:
        plt.yscale('log')
        plt.ylabel('Learning Rate (log scale)')
    
    # 设置x轴格式，避免显示过多刻度
    max_batches = len(batch_lrs)
    if max_batches > 1000:
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate schedule plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"✅ 配置文件已加载: {config_path}")
    return config

# ===============================================================
# 主函数（带配置）
# ===============================================================
def main_with_config(config_path="./config/seq2seq_config.yaml"):
    # 读取配置
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    # === 数据加载 ===
    tokenizer_src, tokenizer_tgt, src_train, tgt_train, src_test, tgt_test = get_tokenizer_data(
        dir_path=cfg["dataset_dir"],
        load_with_local_json=cfg.get("load_with_local_json", False)
    )

    dataset_train = TranslationDataset(src_train, tgt_train, tokenizer_src, tokenizer_tgt, max_len=cfg["max_len"])
    dataset_test = TranslationDataset(src_test, tgt_test, tokenizer_src, tokenizer_tgt, max_len=cfg["max_len"])

    trainloader = DataLoader(dataset_train, batch_size=cfg["batch_size"], shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=cfg["test_batch_size"], shuffle=False)

    print(f"📘 Train size: {len(dataset_train)}, Test size: {len(dataset_test)}")
    print(f"🧠 Vocab (src/tgt): {tokenizer_src.get_vocab_size()} / {tokenizer_tgt.get_vocab_size()}")

    # === 模型初始化 ===
    device = cfg["device"]
    model = Encoder_Decoder_Transformer(
        src_vocab=tokenizer_src.get_vocab_size(),
        tgt_vocab=tokenizer_tgt.get_vocab_size(),
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"]
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("<pad>"))
    param_groups = get_param_groups(model, weight_decay=cfg["weight_decay"])
    optimizer = AdamW(param_groups, lr=cfg["base_lr"], betas=(0.9, 0.98), eps=1e-9)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: transformer_lr(step, cfg["d_model"]))

    print(f"🚀 开始训练: {cfg['experiment_name']}")
    history = train_model(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=cfg["epochs"],
        model_save_dir=cfg["model_save_dir"],
        results_save_dir=cfg["results_save_dir"]
    )

    # === 绘图 ===
    plot_batch_loss(
        history_dict=history,
        save_path=os.path.join(cfg["results_save_dir"], 'batch_loss.png'),
        smoothing=0.95
    )

    plot_learning_rate_schedule(
        history_dict=history,
        save_path=os.path.join(cfg["results_save_dir"], 'batch_lr.png'),
    )

    # === 测试样例生成 ===
    test_text = cfg.get("test_sentence", "I love deep learning.")
    result = generate_text(model, test_text, tokenizer_src, tokenizer_tgt, device)
    print(f"\n🧩 示例输入: {test_text}\n🔤 模型输出: {result}")

# ===============================================================
# CLI 入口
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Seq2Seq model with config file")
    parser.add_argument("--config", type=str, default="./config/seq2seq_config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    main_with_config(args.config)