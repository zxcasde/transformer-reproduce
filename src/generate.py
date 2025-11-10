import torch
from transformer_h import Encoder_Decoder_Transformer
from datasets import load_dataset
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import os
import argparse

def get_tokenizer_data():
    """获取tokenizer和测试数据"""
    tokenizer_src = Tokenizer.from_file("src/src_tokenizer.json")
    tokenizer_tgt = Tokenizer.from_file("src/tgt_tokenizer.json")

    dataset = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-de", cache_dir='/data/yangguang/LLM/data')

    corpus_src_test, corpus_tgt_test = [], []
    for item in dataset['test']['translation']:
        corpus_src_test.append(item['en'])
        corpus_tgt_test.append(item['de'])

    return tokenizer_src, tokenizer_tgt, corpus_src_test, corpus_tgt_test

def create_padding_mask(seq, pad_id):
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)

def generate_square_subsequent_mask(sz):
    return torch.tril(torch.ones(sz, sz)).bool()

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

def load_best_model(model_path, tokenizer_src, tokenizer_tgt, device):
    """加载最佳模型"""
    model = Encoder_Decoder_Transformer(
        src_vocab=tokenizer_src.get_vocab_size(),
        tgt_vocab=tokenizer_tgt.get_vocab_size(),
        d_model=512,
        n_heads=8,
        d_ff=2048,
        num_layers=4
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def plot_sample_predictions(model, test_samples, tokenizer_src, tokenizer_tgt, device, save_path=None):
    """
    绘制样本预测结果 - 四个例子，2x2布局
    """
    # 设置绘图样式
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 生成预测
    predictions = []
    for src_text, tgt_text in test_samples:
        pred_text = generate_text(model, src_text, tokenizer_src, tokenizer_tgt, device)
        predictions.append(pred_text)
    
    # 创建2x2网格
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 定义颜色
    colors = {
        'title': '#2E86AB',
        'source': '#A23B72', 
        'reference': '#18A558',
        'translation': '#C73E1D',
        'bg': '#F8F9FA'
    }
    
    # 绘制每个样本
    for idx, ((src_text, tgt_text), pred_text, ax) in enumerate(zip(test_samples, predictions, axes)):
        # 设置背景色
        ax.set_facecolor(colors['bg'])
        
        # 样本标题
        ax.text(0.5, 0.92, f"Example {idx+1}", fontsize=14, fontweight='bold', 
                color=colors['title'], ha='center', transform=ax.transAxes)
        
        # 源文本
        ax.text(0.05, 0.75, "Source (EN):", fontsize=12, fontweight='bold', 
                color=colors['source'], transform=ax.transAxes)
        ax.text(0.05, 0.65, f'"{src_text}"', fontsize=11, 
                transform=ax.transAxes, style='italic', wrap=True)
        
        # 参考翻译
        ax.text(0.05, 0.50, "Reference (DE):", fontsize=12, fontweight='bold', 
                color=colors['reference'], transform=ax.transAxes)
        ax.text(0.05, 0.40, f'"{tgt_text}"', fontsize=11, 
                transform=ax.transAxes, wrap=True)
        
        # 模型翻译
        ax.text(0.05, 0.25, "Translation (DE):", fontsize=12, fontweight='bold', 
                color=colors['translation'], transform=ax.transAxes)
        ax.text(0.05, 0.15, f'"{pred_text}"', fontsize=11, 
                transform=ax.transAxes, wrap=True)
        
        # 设置坐标轴和边框
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 添加细边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#DDDDDD')
            spine.set_linewidth(1)
    
    # 添加整体标题
    # plt.suptitle("Machine Translation Examples - EN to DE", 
    #             fontsize=18, fontweight='bold', y=0.98)
    
    # 添加副标题
    # plt.figtext(0.5, 0.94, "Comparison of source text, reference translation, and model prediction", 
    #            fontsize=12, ha='center', style='italic', color='#666666')
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"样本预测图已保存至: {save_path}")
    
    plt.show()
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='使用最佳模型生成翻译示例')
    parser.add_argument('--model_path', type=str, required=True, help='最佳模型路径')
    parser.add_argument('--save_path', type=str, default='./sample_predictions.png', help='图片保存路径')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    tokenizer_src, tokenizer_tgt, corpus_src_test, corpus_tgt_test = get_tokenizer_data()
    
    # 选择四个简单的测试样本
    test_samples = [
        ("Hello, how are you today?", "Hallo, wie geht es dir heute?"),
        ("I want to learn German.", "Ich möchte Deutsch lernen."),
        ("The weather is very nice.", "Das Wetter ist sehr schön."),
        ("What time is it now?", "Wie spät ist es jetzt?")
    ]
    
    # 加载模型
    print("加载模型...")
    model = load_best_model(args.model_path, tokenizer_src, tokenizer_tgt, device)
    
    # 生成并展示预测
    print("生成预测...")
    predictions = plot_sample_predictions(
        model, test_samples, tokenizer_src, tokenizer_tgt, device, 
        save_path=args.save_path
    )
    
    # 在控制台也输出结果
    print("\n" + "="*60)
    print("样本预测结果:")
    print("="*60)
    for i, ((src_text, tgt_text), pred_text) in enumerate(zip(test_samples, predictions), 1):
        print(f"\n示例 {i}:")
        print(f"  原文: {src_text}")
        print(f"  参考: {tgt_text}")
        print(f"  翻译: {pred_text}")
    
    print(f"\n完成! 图片已保存至: {args.save_path}")

if __name__ == '__main__':
    main()