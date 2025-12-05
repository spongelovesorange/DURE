# preprocessing/generate_embeddings/text_encoder.py

import torch
import numpy as np
from tqdm import tqdm
import time
from transformers import AutoTokenizer, AutoModel
import os # 导入 os
import sys # 导入 sys

# ✅ (核心修改) 从父目录导入共享函数
try:
    # 添加父目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 从 utils 导入需要的函数 (可能不需要全部导入)
    # from utils import clean_text # 如果需要内部清理
except ImportError as e:
    print(f"导入错误: {e}")
    print("错误: 无法从父目录 (preprocessing/) 导入 utils.py。")
    sys.exit(1)

# 🚨 (移除) 不再需要在这里定义 load_json, clean_text, set_device 等

def generate_local_text(args, item_text_list) -> np.ndarray:
    """使用本地 Transformer 模型生成文本嵌入"""
    print(f"🔹 使用本地模型生成文本嵌入: {args.model_name_or_path}")
    
    # 确保 device 来自 args
    device = getattr(args, 'device', torch.device('cpu')) 
    
    # 优先尝试 HuggingFace 本地模型；失败则回退到 TF-IDF + SVD（完全离线、无额外依赖）
    hf_ok = True
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, cache_dir=args.model_cache_dir
        )
        model = AutoModel.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, cache_dir=args.model_cache_dir
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"[WARN] 加载本地 HuggingFace 模型失败，将回退至 TF-IDF+SVD：{e}")
        hf_ok = False
    
    # (数据准备逻辑 - 保持不变或根据需要调整)
    items, texts = zip(*item_text_list)
    max_item_id = max(items) if items else -1
    order_texts = [[""]] * (max_item_id + 1)
    for item, text in zip(items, texts):
        order_texts[item] = text if text else [""]
    for i in range(len(order_texts)):
        if not order_texts[i]: order_texts[i] = [""] 
    final_texts = [" ".join(t) for t in order_texts]

    if hf_ok:
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(final_texts), args.batch_size), desc="Local Text Encoding (HF)"):
                batch_texts = final_texts[i : i + args.batch_size]
                batch_texts = [t if t.strip() else "N/A" for t in batch_texts] 

                try:
                    encoded = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=args.max_sent_len
                    ).to(device)
                    outputs = model(**encoded)
                    attn = encoded['attention_mask'].unsqueeze(-1)
                    masked = outputs.last_hidden_state * attn
                    mean_output = masked.sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)
                    embeddings.append(mean_output.cpu())
                except Exception as e:
                    print(f"\n[警告] 本地编码批次 {i//args.batch_size} 失败: {e}")
                    emb_dim = getattr(getattr(model, 'config', None), 'hidden_size', 768)
                    embeddings.append(torch.zeros((len(batch_texts), emb_dim)))

        if not embeddings:
            print("[WARN] HF 编码完全失败，启用 TF-IDF+SVD 回退方案。")
            hf_ok = False
        else:
            embeddings = torch.cat(embeddings, dim=0).numpy().astype(np.float32)

    if not hf_ok:
        # 回退：TF-IDF + SVD（密集向量），完全离线，无需 PIL/openai/protobuf/sentencepiece
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            print("[INFO] 使用 TF-IDF + TruncatedSVD 生成文本嵌入 (离线回退方案)...")
            texts = [t if isinstance(t, str) and t.strip() else "N/A" for t in final_texts]
            tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
            X = tfidf.fit_transform(texts)
            n_components = min(1024, X.shape[1]-1) if X.shape[1] > 1 else 1
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            X_reduced = svd.fit_transform(X)
            embeddings = X_reduced.astype(np.float32)
            print(f"[INFO] TF-IDF+SVD 嵌入维度: {embeddings.shape}")
        except Exception as e:
            raise RuntimeError(f"离线回退(TFIDF+SVD)失败: {e}")
    
    # 验证数量
    if embeddings.shape[0] != len(final_texts):
         print(f"[警告] 本地文本嵌入数量 ({embeddings.shape[0]}) 与预期 ({len(final_texts)}) 不符！")
         # 填充或截断以匹配
         target_len = len(final_texts)
         current_len = embeddings.shape[0]
         emb_dim = embeddings.shape[1]
         if current_len < target_len: # 填充
              print(" -> 将用零向量填充。")
              padding = np.zeros((target_len - current_len, emb_dim), dtype=np.float32)
              embeddings = np.concatenate([embeddings, padding], axis=0)
         else: # 截断
              print(" -> 将截断多余部分。")
              embeddings = embeddings[:target_len]

    print(f"本地文本嵌入维度: {embeddings.shape}")
    return embeddings

def generate_api_text(args, item_text_list) -> np.ndarray:
    """使用 OpenAI API 生成文本嵌入"""
    print(f"🔹 使用 API 模型生成文本嵌入: {args.sent_emb_model}")
    try:
        from openai import OpenAI
    except ImportError:
        print("错误: 'openai' 库未找到。请运行: pip install openai")
        raise # 重新抛出，让主脚本知道依赖缺失

    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)

    # (数据准备逻辑 - 保持不变)
    items, texts = zip(*item_text_list)
    max_item_id = max(items) if items else -1
    order_texts = [[""]] * (max_item_id + 1)
    for item, text in zip(items, texts):
        order_texts[item] = text if text else [""]
    for i in range(len(order_texts)):
        if not order_texts[i]: order_texts[i] = [""] 
    final_texts = [" ".join(t) for t in order_texts]

    sent_embs = []
    api_emb_dim = args.api_emb_dim 
    if api_emb_dim <= 0: # 尝试根据模型名猜测
        if 'large' in args.sent_emb_model: api_emb_dim = 3072
        elif 'small' in args.sent_emb_model: api_emb_dim = 1536
        else: api_emb_dim = 0 # 无法猜测时保持 0
        
    print(f"[INFO] 预期/猜测的 API 维度: {api_emb_dim if api_emb_dim > 0 else '自动检测'}")

    for i in tqdm(range(0, len(final_texts), args.batch_size), desc="API Text Encoding"):
        batch = final_texts[i : i + args.batch_size]
        batch = [t if t.strip() else "N/A" for t in batch]
        
        try:
            response = client.embeddings.create(model=args.sent_emb_model, input=batch)
            batch_embeddings = [np.array(d.embedding, dtype=np.float32) for d in response.data] # 直接转 numpy
            sent_embs.extend(batch_embeddings)
            
            if api_emb_dim <= 0 and batch_embeddings:
                api_emb_dim = len(batch_embeddings[0])
                print(f"\n[INFO] 实际检测到 API 嵌入维度为: {api_emb_dim}")
                
        except Exception as e:
            print(f"\n[警告] API 请求批次 {i//args.batch_size} 失败: {e}")
            if api_emb_dim <= 0:
                 print("错误：API 维度未知且未在 --api_emb_dim 中指定，无法创建零向量。")
                 api_emb_dim = 1024 # 最后的默认回退
                 print(f"警告：将假设维度为 {api_emb_dim}。")
                 
            sent_embs.extend([np.zeros(api_emb_dim, dtype=np.float32) for _ in batch])
            time.sleep(1)

    if not sent_embs: # 处理完全失败
         raise RuntimeError("未能生成任何 API 文本嵌入。")

    # 尝试将 list of numpy arrays 转换为单个 numpy array
    try:
        sent_embs = np.stack(sent_embs, axis=0)
    except ValueError as e:
         print(f"错误：无法将 API 返回的嵌入堆叠成数组 ({e})。可能维度不一致。")
         # 尝试找出不一致的维度
         dims = [emb.shape for emb in sent_embs if isinstance(emb, np.ndarray)]
         print(f"检测到的维度: {set(dims)}")
         # 选择填充或报错，这里报错
         raise RuntimeError("API 返回的嵌入维度不一致。") from e
         
    args.api_emb_dim = api_emb_dim # 更新 args

    # 验证数量
    if sent_embs.shape[0] != len(final_texts):
         print(f"[警告] API 输出嵌入数量 ({sent_embs.shape[0]}) 与预期 ({len(final_texts)}) 不符！")
         # 填充或截断
         target_len = len(final_texts)
         current_len = sent_embs.shape[0]
         emb_dim = sent_embs.shape[1]
         if current_len < target_len:
              print(" -> 将用零向量填充。")
              padding = np.zeros((target_len - current_len, emb_dim), dtype=np.float32)
              sent_embs = np.concatenate([sent_embs, padding], axis=0)
         else:
              print(" -> 将截断多余部分。")
              sent_embs = sent_embs[:target_len]

    print(f"API 文本嵌入维度: {sent_embs.shape}")
    return sent_embs