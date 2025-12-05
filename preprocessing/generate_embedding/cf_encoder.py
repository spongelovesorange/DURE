# preprocessing/generate_embeddings/cf_encoder.py

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# ✅ (核心修改) 从父目录导入共享函数
try:
    # 添加父目录 (preprocessing/) 到 Python 路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import load_json, set_device # 导入需要的函数
    print("[INFO] cf_encoder: 成功从父目录 utils.py 导入共享函数。")
except ImportError as e:
    print(f"导入错误: {e}")
    print("错误: 无法从父目录 (preprocessing/) 导入 utils.py。请检查文件结构。")
    sys.exit(1)

# 🚨 (移除) 不再需要导入或定义 common_utils 中的函数
# try:
#     from .common_utils import build_output_path 
# except ImportError: ...

# 🚨 (移除) 不再需要在这里重新定义 load_json, set_device

# =================================================================
# ================== SASRec 数据集和模型 (保持不变) =============
# =================================================================

class SASRecDataset(Dataset):
    # ... (代码保持不变) ...
    def __init__(self, data_path, max_seq_len): 
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.lines = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f: # 添加 encoding
                for line in f:
                    self.lines.append(line.strip())
        except FileNotFoundError:
             print(f"错误：SASRec 训练文件未找到: {data_path}")
             raise # 重新抛出，让调用者知道
        except Exception as e:
             print(f"错误：读取 SASRec 训练文件失败: {e}")
             raise

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        try:
            line = self.lines[idx]
            data = json.loads(line)
            history_ids = [int(i) + 1 for i in data["history"]] 
            target_id = int(data["target"]) + 1 
            seq = history_ids[-self.max_seq_len:]
            seq_len = len(seq)
            padding_len = self.max_seq_len - seq_len
            seq = seq + [0] * padding_len
            return torch.tensor(seq, dtype=torch.long), \
                   torch.tensor(target_id, dtype=torch.long), \
                   torch.tensor(seq_len, dtype=torch.long)
        except json.JSONDecodeError:
             print(f"警告：解析第 {idx} 行 JSON 失败: {line}")
             # 返回一个占位符或引发异常，这里选择占位符，但可能导致训练问题
             # 更健壮的方式是在加载时过滤掉无效行
             return torch.zeros(self.max_seq_len, dtype=torch.long), \
                    torch.tensor(0, dtype=torch.long), \
                    torch.tensor(0, dtype=torch.long)
        except KeyError as e:
            print(f"警告：第 {idx} 行缺少键 {e}: {line}")
            return torch.zeros(self.max_seq_len, dtype=torch.long), \
                   torch.tensor(0, dtype=torch.long), \
                   torch.tensor(0, dtype=torch.long)


class SASRecModel(nn.Module):
    # ... (代码保持不变) ...
    def __init__(self, n_items, hidden_dim, max_seq_len, n_layers, n_heads, dropout=0.1):
        super(SASRecModel, self).__init__()
        self.n_items = n_items; self.hidden_dim = hidden_dim
        self.item_embedding = nn.Embedding(self.n_items + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True # 添加 norm_first
        )
        # 添加 LayerNorm
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=encoder_norm) # 应用 norm
        self.layer_norm = nn.LayerNorm(hidden_dim) # 输出前的 LayerNorm

    def forward(self, item_seq, seq_lengths):
        # 边界检查 seq_lengths
        seq_lengths = torch.clamp(seq_lengths, min=1) # 确保最小长度为 1
        last_item_indices = seq_lengths - 1 # 现在安全了

        item_emb = self.item_embedding(item_seq)
        pos_ids = torch.arange(item_seq.size(1), device=item_seq.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        x = self.emb_dropout(item_emb + pos_emb)
        padding_mask = (item_seq == 0)
        # 确保 mask 和 input 维度兼容
        transformer_out = self.transformer_encoder(x, src_key_padding_mask=padding_mask) 
        transformer_out = self.layer_norm(transformer_out) # 应用输出 LayerNorm
        
        batch_indices = torch.arange(transformer_out.size(0), device=transformer_out.device)
        # last_item_indices = seq_lengths - 1 # 移到前面
        last_item_emb = transformer_out[batch_indices, last_item_indices, :]
        logits = last_item_emb @ self.item_embedding.weight.T
        return logits

# =================================================================
# ================== 主训练与提取函数 (保持不变) ==================
# =================================================================

def train_and_extract_sasrec(args, n_items: int) -> np.ndarray:
    """
    训练 SASRec 模型并提取物品嵌入。
    (函数体保持不变，因为它已经依赖于从 utils 导入的函数)
    """
    print(f"🔹 使用 SASRec 训练协同过滤嵌入...")
    device = args.device # device 已由 main_generate 设置

    # --- 1. 构建路径 ---
    data_dir = os.path.join(args.save_root, args.dataset) 
    train_path = os.path.join(data_dir, f"{args.dataset}.train.jsonl")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"错误: 找不到 SASRec 训练文件 {train_path}")

    # --- 2. 创建 Dataset 和 DataLoader ---
    print("加载 SASRec 训练数据...")
    try:
        # 使用 getattr 安全获取 num_workers
        num_workers = getattr(args, 'num_workers', 0) 
        train_dataset = SASRecDataset(train_path, args.sasrec_max_seq_len) 
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, 
            shuffle=True, num_workers=num_workers, 
            pin_memory=(device.type == 'cuda')
        )
    except Exception as e:
        print(f"创建 SASRec DataLoader 失败: {e}")
        raise

    # --- 3. 初始化模型、损失和优化器 ---
    try:
        model = SASRecModel(
            n_items=n_items,
            hidden_dim=args.sasrec_hidden_dim,
            max_seq_len=args.sasrec_max_seq_len,
            n_layers=args.sasrec_n_layers,
            n_heads=args.sasrec_n_heads,
            dropout=args.sasrec_dropout
        ).to(device)
    except Exception as e:
         print(f"初始化 SASRec 模型失败: {e}")
         raise
    
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 忽略 PAD 目标 (虽然理论上 target 不应为 0)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.sasrec_lr, 
        weight_decay=args.sasrec_weight_decay
    )
    
    print("开始训练 SASRec...")
    start_time = time.time()
    
    for epoch in range(1, args.sasrec_epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"SASRec Epoch {epoch}/{args.sasrec_epochs}")
        
        for batch in pbar:
            try:
                seq, target, seq_len = [b.to(device) for b in batch]
                
                # 跳过 seq_len 为 0 的无效样本 (由 Dataset 返回的占位符导致)
                valid_mask = (seq_len > 0)
                if not valid_mask.any(): continue # 如果整个批次都无效
                
                seq = seq[valid_mask]
                target = target[valid_mask]
                seq_len = seq_len[valid_mask]

                optimizer.zero_grad()
                logits = model(seq, seq_len)
                loss = criterion(logits, target)
                
                # 检查 loss 是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                     print(f"\n[警告] Epoch {epoch}: 检测到无效 Loss 值 ({loss.item()})，跳过此批次。")
                     continue # 跳过无效批次

                loss.backward()
                # (可选) 梯度裁剪
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"Loss": loss.item()})
            except Exception as e:
                 print(f"\n[警告] 训练批次中发生错误: {e}")
                 # 可以选择 continue 跳过批次，或 raise 终止训练
                 continue # 暂时跳过

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch} 完成. 平均损失: {avg_loss:.4f}")

    train_duration = time.time() - start_time
    print(f"SASRec 训练完成. 耗时: {train_duration:.2f} 秒.")
    
    # --- 4. 提取嵌入 ---
    print("正在提取物品嵌入...")
    model.eval()
    
    try:
        with torch.no_grad():
            # 提取整个嵌入矩阵 [n_items+1, D]
            embeddings = model.item_embedding.weight.data.cpu().numpy()
        
        # 移除 0 号索引的 [PAD] 嵌入
        embeddings = embeddings[1:] # [n_items, D]
        
        print(f"提取的 SASRec 嵌入维度: {embeddings.shape}")
        
        # 验证数量
        if embeddings.shape[0] != n_items:
             print(f"[警告] 提取的嵌入数量 ({embeddings.shape[0]}) 与预期物品数量 ({n_items}) 不符！")
             # 尝试修复：如果数量少于预期，用零填充
             if embeddings.shape[0] < n_items:
                  print(f" -> 将用零向量填充至 {n_items} 个。")
                  padding = np.zeros((n_items - embeddings.shape[0], embeddings.shape[1]), dtype=embeddings.dtype)
                  embeddings = np.concatenate([embeddings, padding], axis=0)
             else: # 如果多于预期（理论上不应发生），截断
                  print(f" -> 将截断至 {n_items} 个。")
                  embeddings = embeddings[:n_items]
        
        return embeddings.astype(np.float32)
        
    except Exception as e:
         print(f"提取 SASRec 嵌入失败: {e}")
         raise # 重新抛出

# 🚨 (移除) 不再需要 main 或 argparse
# if __name__ == "__main__":
#     args = ...
#     train_and_extract_sasrec(...)