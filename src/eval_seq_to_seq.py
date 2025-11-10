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
import random
import numpy as np
import math
import argparse
from pathlib import Path

def get_tokenizer_data(dataset_name=''):
    tokenizer_src = Tokenizer.from_file("src/src_tokenizer.json")
    tokenizer_tgt = Tokenizer.from_file("src/tgt_tokenizer.json")

    dataset = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-de", cache_dir='/data/yangguang/LLM/data')

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
def batched_beam_search_decode(
    model,
    src,
    src_mask,
    tokenizer_tgt,
    device,
    beam_size=5,
    max_len=128,
    length_penalty=0.7
):
    """批量 beam search 解码"""
    model.eval()
    batch_size = src.size(0)
    bos_id = tokenizer_tgt.token_to_id("<sos>")
    eos_id = tokenizer_tgt.token_to_id("<eos>")

    sequences = torch.full((batch_size, beam_size, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.zeros(batch_size, beam_size, device=device)
    finished = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)

    for step in range(max_len - 1):
        current_beam = sequences.size(1)
        input_tgt = sequences.view(batch_size * current_beam, -1)

        tgt_mask = generate_square_subsequent_mask(input_tgt.size(1)).to(device)
        src_rep = src.unsqueeze(1).expand(-1, current_beam, -1).reshape(batch_size * current_beam, -1)
        src_mask_rep = src_mask.unsqueeze(1).expand(-1, current_beam, -1, -1, -1).reshape(batch_size * current_beam, 1, 1, -1)

        logits = model(src_rep, input_tgt, src_mask_rep, tgt_mask)
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
        log_probs = log_probs.view(batch_size, current_beam, -1)

        total_scores = scores.unsqueeze(-1) + log_probs
        topk_scores, topk_indices = total_scores.view(batch_size, -1).topk(beam_size, dim=-1)

        beam_indices = topk_indices // log_probs.size(-1)
        token_indices = topk_indices % log_probs.size(-1)

        new_sequences, new_finished = [], []
        for b in range(batch_size):
            seqs = sequences[b][beam_indices[b]]
            next_tokens = token_indices[b].unsqueeze(-1)
            new_seq = torch.cat([seqs, next_tokens], dim=-1)
            new_sequences.append(new_seq)
            new_finished.append(finished[b][beam_indices[b]] | (token_indices[b] == eos_id))

        sequences = torch.stack(new_sequences, dim=0)
        finished = torch.stack(new_finished, dim=0)
        scores = topk_scores

        if finished.all():
            break

    best_indices = scores.argmax(dim=-1)
    best_sequences = [sequences[b, best_indices[b]].tolist() for b in range(batch_size)]
    return best_sequences

@torch.no_grad()
def beam_decode_batch(model, src, src_mask, tokenizer_tgt, device, beam_size=5, max_len=128):
    """对一个 batch 使用 beam search 解码"""
    seqs = batched_beam_search_decode(model, src, src_mask, tokenizer_tgt, device, beam_size, max_len)
    max_len_out = max(len(s) for s in seqs)
    pad_id = tokenizer_tgt.token_to_id("<pad>")
    padded = [s + [pad_id]*(max_len_out - len(s)) for s in seqs]
    return torch.tensor(padded, dtype=torch.long, device=device)

@torch.no_grad()
def evaluate_bleu(model, dataloader, tokenizer_src, tokenizer_tgt, device, max_len=128, max_batches=None):
    """评估BLEU分数"""
    refs, hyps = [], []
    model.eval()

    pad_id = tokenizer_src.token_to_id("<pad>")
    smooth_fn = SmoothingFunction().method4

    for idx, (src, tgt) in enumerate(tqdm(dataloader, desc="Evaluating (BLEU)")):
        if (max_batches is not None) and (idx >= max_batches):
            break

        src, tgt = src.to(device), tgt.to(device)
        src_mask = create_padding_mask(src, pad_id).to(device)

        pred_batch = beam_decode_batch(model, src, src_mask, tokenizer_tgt, device, beam_size=6, max_len=max_len)

        for i in range(src.size(0)):
            pred_text = tokenizer_tgt.decode([int(x) for x in pred_batch[i].tolist()], skip_special_tokens=True)
            tgt_text  = tokenizer_tgt.decode([int(x) for x in tgt[i].tolist()], skip_special_tokens=True)

            pred_text = pred_text.replace("<sos>", "").replace("<eos>", "").replace("<pad>", "").strip()
            tgt_text  = tgt_text.replace("<sos>", "").replace("<eos>", "").replace("<pad>", "").strip()

            if len(pred_text) == 0 or len(tgt_text) == 0:
                continue

            refs.append([tgt_text.split()])
            hyps.append(pred_text.split())

    if len(hyps) == 0:
        print("No hypotheses collected for BLEU evaluation.")
        return 0.0

    bleu = corpus_bleu(refs, hyps, smoothing_function=smooth_fn)
    return bleu

@torch.no_grad()
def eval_loss_func(model, dataloader, criterion, tokenizer_src, tokenizer_tgt, device, max_batches=50):
    """评估损失和计算PPL"""
    model.eval()
    pad_id = tokenizer_src.token_to_id("<pad>")
    total_loss = 0.0
    count = 0
    
    for i, (src, tgt) in enumerate(dataloader):
        if i >= max_batches: 
            break
        src, tgt = src.to(device), tgt.to(device)
        src_mask = create_padding_mask(src, pad_id).to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        preds = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(preds.reshape(-1, preds.size(-1)), tgt_output.reshape(-1))
        total_loss += loss.item() * tgt_output.numel()
        count += tgt_output.numel()
    
    avg_loss = total_loss / count
    ppl = math.exp(avg_loss)  # 计算困惑度
    return avg_loss, ppl

@torch.no_grad()
def generate_text(model, src_text, tokenizer_src, tokenizer_tgt, device, max_len=128):
    """生成文本"""
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
        
        if next_token.item() == eos_id:
            break
    
    output_ids = ys.squeeze(0).tolist()
    output_text = tokenizer_tgt.decode(output_ids, skip_special_tokens=True)
    return output_text.strip()

def setup_plot_style():
    """设置绘图样式"""
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['font.family'] = 'DejaVu Sans'

def plot_evaluation_metrics(evaluation_results, save_path=None):
    """绘制评估指标变化图"""
    setup_plot_style()
    
    epochs = [result['epoch'] for result in evaluation_results]
    eval_losses = [result['eval_loss'] for result in evaluation_results]
    ppls = [result['ppl'] for result in evaluation_results]
    bleus = [result['bleu'] * 100 for result in evaluation_results]  # 转换为百分比
    
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # 绘制eval_loss
    ax1.plot(epochs, eval_losses, 'b-o', linewidth=2, markersize=6)
    ax1.set_title('Evaluation Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # 绘制PPL（使用对数刻度）
    ax2.plot(epochs, ppls, 'r-o', linewidth=2, markersize=6)
    ax2.set_title('Perplexity (PPL) over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PPL')
    ax2.set_yscale('log')  # 使用对数刻度，因为PPL通常很大
    ax2.grid(True, alpha=0.3)
    
    # 绘制BLEU
    ax3.plot(epochs, bleus, 'g-o', linewidth=2, markersize=6)
    ax3.set_title('BLEU Score over Epochs')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('BLEU (%)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"评估指标图已保存至: {save_path}")
    
    plt.show()

def find_best_model(evaluation_results, metric='bleu', higher_is_better=True):
    """根据指定指标找到最佳模型"""
    if higher_is_better:
        best_result = max(evaluation_results, key=lambda x: x[metric])
    else:
        best_result = min(evaluation_results, key=lambda x: x[metric])
    
    return best_result

def evaluate_all_models(model_dir, test_loader, tokenizer_src, tokenizer_tgt, device, max_epochs=50, max_batches=20):
    """评估所有保存的模型"""
    evaluation_results = []
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("<pad>"))
    
    # 获取所有模型文件
    model_files = []
    for file in Path(model_dir).glob("checkpoint_epoch_*.pth"):
        try:
            epoch = int(file.stem.split('_')[-1])
            model_files.append((epoch, file))
        except:
            continue
    
    # 按epoch排序
    model_files.sort(key=lambda x: x[0])
    
    # 限制评估的epoch数量
    if max_epochs:
        model_files = model_files[:max_epochs]
    
    print(f"找到 {len(model_files)} 个模型文件进行评估")
    
    for epoch, model_file in tqdm(model_files, desc="评估所有模型"):
        try:
            # 加载模型
            model = Encoder_Decoder_Transformer(
                src_vocab=tokenizer_src.get_vocab_size(),
                tgt_vocab=tokenizer_tgt.get_vocab_size(),
                d_model=512,
                n_heads=8,
                d_ff=2048,
                num_layers=4
            ).to(device)
            
            checkpoint = torch.load(model_file, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 评估指标
            eval_loss, ppl = eval_loss_func(model, test_loader, criterion, tokenizer_src, tokenizer_tgt, device, max_batches)
            bleu = evaluate_bleu(model, test_loader, tokenizer_src, tokenizer_tgt, device, max_batches=max_batches)
            
            result = {
                'epoch': epoch,
                'eval_loss': eval_loss,
                'ppl': ppl,
                'bleu': bleu,
                'model_path': str(model_file)
            }
            
            evaluation_results.append(result)
            
            print(f"Epoch {epoch}: Loss={eval_loss:.4f}, PPL={ppl:.2f}, BLEU={bleu*100:.2f}%")
            
        except Exception as e:
            print(f"评估 epoch {epoch} 时出错: {e}")
            continue
    
    return evaluation_results

def main():
    parser = argparse.ArgumentParser(description='评估保存的翻译模型')
    parser.add_argument('--model_dir', type=str, default='./models', help='模型保存目录')
    parser.add_argument('--max_epochs', type=int, default=50, help='最大评估epoch数')
    parser.add_argument('--max_batches', type=int, default=20, help='每个评估的最大batch数')
    parser.add_argument('--results_dir', type=str, default='./results/seq_to_seq/evaluation_results', help='结果保存目录')
    
    args = parser.parse_args()
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    tokenizer_src, tokenizer_tgt, _, _, corpus_src_test, corpus_tgt_test = get_tokenizer_data()
    dataset_test = TranslationDataset(corpus_src_test, corpus_tgt_test, tokenizer_src, tokenizer_tgt)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    
    # 评估所有模型
    print("开始评估所有模型...")
    evaluation_results = evaluate_all_models(
        args.model_dir, test_loader, tokenizer_src, tokenizer_tgt, device,
        max_epochs=args.max_epochs, max_batches=args.max_batches
    )
    
    # 保存评估结果
    results_file = os.path.join(args.results_dir, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
    print(f"评估结果已保存至: {results_file}")
    
    # 绘制评估指标图
    plot_file = os.path.join(args.results_dir, 'evaluation_metrics.png')
    plot_evaluation_metrics(evaluation_results, save_path=plot_file)
    
    # 找到最佳模型
    best_by_bleu = find_best_model(evaluation_results, metric='bleu', higher_is_better=True)
    best_by_loss = find_best_model(evaluation_results, metric='eval_loss', higher_is_better=False)
    best_by_ppl = find_best_model(evaluation_results, metric='ppl', higher_is_better=False)
    
    print("\n" + "="*50)
    print("最佳模型结果:")
    print(f"按BLEU选择 - Epoch {best_by_bleu['epoch']}: BLEU={best_by_bleu['bleu']*100:.2f}%, Loss={best_by_bleu['eval_loss']:.4f}, PPL={best_by_bleu['ppl']:.2f}")
    print(f"按Loss选择 - Epoch {best_by_loss['epoch']}: BLEU={best_by_loss['bleu']*100:.2f}%, Loss={best_by_loss['eval_loss']:.4f}, PPL={best_by_loss['ppl']:.2f}")
    print(f"按PPL选择 - Epoch {best_by_ppl['epoch']}: BLEU={best_by_ppl['bleu']*100:.2f}%, Loss={best_by_ppl['eval_loss']:.4f}, PPL={best_by_ppl['ppl']:.2f}")
    print("="*50)
    
    # 使用最佳BLEU模型进行生成示例
    print("\n使用最佳BLEU模型进行生成示例:")
    best_model_path = best_by_bleu['model_path']
    
    # 加载最佳模型
    model = Encoder_Decoder_Transformer(
        src_vocab=tokenizer_src.get_vocab_size(),
        tgt_vocab=tokenizer_tgt.get_vocab_size(),
        d_model=512,
        n_heads=8,
        d_ff=2048,
        num_layers=4
    ).to(device)
    
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 生成示例
    test_sentences = [
        "Thank you so much, Chris.",
        "Hello, how are you today?",
        "The weather is very nice today.",
        "I would like to order a coffee.",
        "What time does the train leave?"
    ]
    
    print("\n生成示例:")
    for i, src_text in enumerate(test_sentences, 1):
        translation = generate_text(model, src_text, tokenizer_src, tokenizer_tgt, device)
        print(f"{i}. 原文: {src_text}")
        print(f"   翻译: {translation}")
        print()
    
    # 保存最佳模型信息
    best_model_info = {
        'best_by_bleu': best_by_bleu,
        'best_by_loss': best_by_loss,
        'best_by_ppl': best_by_ppl,
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    best_model_file = os.path.join(args.results_dir, 'best_model_info.json')
    with open(best_model_file, 'w', encoding='utf-8') as f:
        json.dump(best_model_info, f, indent=4, ensure_ascii=False)
    
    print(f"最佳模型信息已保存至: {best_model_file}")
    print("评估完成!")

if __name__ == '__main__':
    main()