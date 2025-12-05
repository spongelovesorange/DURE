import html
import json
import os
import pickle
import re
import time

import torch
# import gensim
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler
import collections
from typing import Any, Dict, List, Optional, Tuple
# OpenAI (optional). Define a fallback to avoid NameError when not installed.
try:
    import openai  # type: ignore
except Exception:  # ModuleNotFoundError or other
    openai = None



def get_res_batch(model_name, prompt_list, max_tokens, api_info):
    """Optional OpenAI helper. Requires 'openai' package and network. Safe no-op offline."""
    if openai is None:
        raise RuntimeError("OpenAI package not available in this environment.")
    while True:
        try:
            res = openai.Completion.create(
                model=model_name,
                prompt=prompt_list,
                temperature=0.4,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            output_list = []
            for choice in res['choices']:
                output = choice['text'].strip()
                output_list.append(output)
            return output_list
        except openai.error.AuthenticationError as e:
            print(e)
            openai.api_key = api_info["api_key_list"].pop()
            time.sleep(10)
        except openai.error.RateLimitError as e:
            print(e)
            if str(e) == "You exceeded your current quota, please check your plan and billing details.":
                openai.api_key = api_info["api_key_list"].pop()
                time.sleep(10)
            else:
                print('\nopenai.error.RateLimitError\nRetrying...')
                time.sleep(10)
        except openai.error.ServiceUnavailableError as e:
            print(e)
            print('\nopenai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(10)
        except openai.error.Timeout:
            print('\nopenai.error.Timeout\nRetrying...')
            time.sleep(10)
        except openai.error.APIError as e:
            print(e)
            print('\nopenai.error.APIError\nRetrying...')
            time.sleep(10)
        except openai.error.APIConnectionError as e:
            print(e)
            print('\nopenai.error.APIConnectionError\nRetrying...')
            time.sleep(10)
        except Exception as e:
            print(e)
            return None




def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')

def load_plm(model_path='bert-base-uncased', kwargs=None):

    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)

    print("Load Model:", model_path)

    model = AutoModel.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    return tokenizer, model

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def clean_text(raw_text):
    if isinstance(raw_text, list):
        new_raw_text=[]
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r'</?\w+[^>]*>', '', raw)
            raw = re.sub(r'["\n\r]*', '', raw)
            new_raw_text.append(raw.strip())
        cleaned_text = ' '.join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r'</?\w+[^>]*>', '', cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters.append(inter)
    return new_inters

def write_json_file(dic, file):
    print('Writing json file: ',file)
    with open(file, 'w') as fp:
        json.dump(dic, fp, indent=4)

def write_remap_index(unit2index, file):
    print('Writing remap file: ',file)
    with open(file, 'w') as fp:
        for unit in unit2index:
            fp.write(unit + '\t' + str(unit2index[unit]) + '\n')

# preprocessing/utils.py (增强版)

import html
import json
import os
import pickle
import re
import time
import argparse # 导入 argparse 用于类型提示
from pathlib import Path # 使用 Path 对象处理路径
import joblib # 用于保存 PCA 模型
import numpy as np

from sklearn.decomposition import PCA
import torch
import collections
# (移除不再需要的 transformers, openai 等特定库导入，这些应在各自 encoder 模块中)

# ====================================================
# ========= 核心通用函数 (来自您原有的 utils.py) =========
# ====================================================

def check_path(path):
    """(保持不变) 确保目录存在"""
    # 使用 Path 对象更佳
    Path(path).mkdir(parents=True, exist_ok=True)

def set_device(gpu_id: int) -> torch.device:
    """(保持不变) 设置 PyTorch 设备"""
    if gpu_id < 0:
        print("[INFO] Using CPU.")
        return torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(device)}")
        return device
    else:
        print("[WARN] CUDA not available, falling back to CPU.")
        return torch.device('cpu')

def load_json(file: str) -> Any:
    """(保持不变) 更健壮地加载 JSON 文件"""
    if not os.path.exists(file):
        print(f"[WARN] JSON file not found: {file}")
        return None
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to decode JSON file: {file}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load JSON file {file}: {e}")
        return None

def clean_text(raw_text: Any) -> str:
    """(保持不变) 清理文本中的 HTML 标签、换行符等"""
    text_to_clean = ""
    if isinstance(raw_text, list):
        text_to_clean = ' '.join(str(item) for item in raw_text)
    elif isinstance(raw_text, dict):
         # 将字典转换为 "key1: value1, key2: value2" 格式 (或根据需要调整)
        text_to_clean = ", ".join(f"{k}: {v}" for k, v in raw_text.items())
    elif isinstance(raw_text, (str, int, float)):
         text_to_clean = str(raw_text)
    else:
         return "" # 对于无法处理的类型返回空

    try:
        cleaned_text = html.unescape(text_to_clean)
        cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text) # 移除 HTML 标签
        cleaned_text = re.sub(r'[\n\r]+', ' ', cleaned_text) # 替换换行符为空格
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text) # 合并多个空格
        cleaned_text = cleaned_text.replace('"', '').strip() # 移除引号并去除首尾空格
        # (移除末尾句点逻辑，可能不需要或过于特定)
        # index = -1 ...
    except Exception as e:
        # print(f"[WARN] Error cleaning text: {raw_text}. Error: {e}")
        cleaned_text = "" # 出错时返回空字符串

    # (移除长度限制逻辑，让调用者决定)
    # if len(cleaned_text) >= 2000: cleaned_text = ''
    return cleaned_text

def load_pickle(filename: str) -> Any:
    """(保持不变) 加载 Pickle 文件"""
    if not os.path.exists(filename):
         print(f"[WARN] Pickle file not found: {filename}")
         return None
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as e:
         print(f"[ERROR] Failed to load pickle file {filename}: {e}")
         return None

def write_json_file(dic: dict, file: str):
    """(保持不变) 写入 JSON 文件"""
    print(f'Writing json file: {file}')
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'w', encoding='utf-8') as fp:
            json.dump(dic, fp, indent=4, ensure_ascii=False) # 使用 ensure_ascii=False 保留非 ASCII 字符
    except Exception as e:
         print(f"[ERROR] Failed to write JSON file {file}: {e}")

def write_remap_index(unit2index: dict, file: str):
    """(保持不变) 写入 remap 文件 (ID 映射)"""
    print(f'Writing remap file: {file}')
    try:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'w', encoding='utf-8') as fp:
            # 按 index 排序写入，保证一致性
            for unit, index in sorted(unit2index.items(), key=lambda item: int(item[1])):
                fp.write(f"{unit}\t{index}\n")
    except Exception as e:
        print(f"[ERROR] Failed to write remap file {file}: {e}")

# ========================================================
# ========= 新增：用于 Embedding 生成的通用函数 =========
# ========================================================

def get_id2item_dict(item2id_file: str) -> Dict[str, str]:
    """
    (新增) 从 .item2id 文件加载 新ID(str) -> 原始ID(str) 的映射。
    """
    if not os.path.exists(item2id_file):
        raise FileNotFoundError(f"item2id 文件未找到: {item2id_file}")
    id2item = {}
    try:
        with open(item2id_file, "r", encoding='utf-8') as fp:
            for line_num, line in enumerate(fp):
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    item, item_id = parts # item 是原始ID, item_id 是新ID (str)
                    id2item[item_id] = item
                elif line.strip(): # 忽略空行，但对格式错误的行发出警告
                     print(f"[WARN] item2id 文件第 {line_num+1} 行格式错误: '{line.strip()}'")
        if not id2item:
            raise RuntimeError(f"未能从 {item2id_file} 加载任何 ID 映射。")
    except Exception as e:
        print(f"[ERROR] 读取 item2id 文件失败 ({item2id_file}): {e}")
        raise # 重新抛出异常，因为这是关键数据
    return id2item

def build_text_map(args: argparse.Namespace, id2item: Dict[str, str], item_meta: Dict[str, Dict]) -> Dict[str, str]:
    """
    (新增) 根据 dataset_type 从 item_meta 构建 原始ID(str) -> 清理后文本(str) 的映射。
    """
    if not item_meta:
         print("[WARN] item_meta 为空，无法构建 text_map。")
         return {}

    # 根据 dataset_type 选择字段
    features = []
    if args.dataset_type == 'movielens':
        features = ['title', 'description', 'genres', 'year'] # 添加 year
    elif args.dataset_type == 'amazon':
        features = ['title', 'description', 'brand', 'categories']
    else:
        print(f"[WARN] 未知的 dataset_type: {args.dataset_type}，将尝试使用通用字段 ['title', 'description']")
        features = ['title', 'description']

    print(f"将使用以下元数据字段构建文本: {features}")
    text_map = {}
    missing_meta_count = 0
    
    # 遍历 id2item 来确保我们只处理有效的 item
    for new_id_str, orig_id in id2item.items():
        # item_meta 的键是新ID(str)
        meta_data = item_meta.get(new_id_str) 
        if not meta_data:
             missing_meta_count += 1
             text_map[orig_id] = "N/A" # 或者 ""
             continue

        parts = []
        for f in features:
            if f in meta_data:
                val = meta_data[f]
                # 对列表特殊处理 (例如 genres)
                if isinstance(val, list):
                    val = ", ".join(clean_text(str(x)) for x in val if str(x).strip()) # 清理列表元素
                else:
                    val = clean_text(str(val)) # 清理普通字段
                
                if val: # 只添加非空字段
                    # 可以考虑添加字段名前缀，例如 "Title: ... Brand: ..."
                    # parts.append(f"{f.capitalize()}: {val}") 
                    parts.append(val)
                    
        text = " ".join(parts).strip()
        text_map[orig_id] = text if text else "N/A" # 保证至少是 "N/A"

    if missing_meta_count > 0:
         print(f"[WARN] {missing_meta_count} 个 item 在 item.json 中缺少元数据。")
         
    return text_map

def build_output_path(args: argparse.Namespace, modality_tag: str, model_tag: str) -> str:
    """(新增) 构建标准化的 embedding 输出路径"""
    emb_dir = os.path.join(args.save_root, args.dataset, "embeddings")
    check_path(emb_dir) # 使用 check_path 确保目录存在
    
    # 清理 model_tag 中的非法字符
    safe_model_tag = model_tag.split('/')[-1].replace('/', '-').replace('\\', '-')
    
    filename = f"{args.dataset}.emb-{modality_tag}-{safe_model_tag}.npy"
    return os.path.join(emb_dir, filename)

def apply_pca_and_save(original_embeddings: np.ndarray, args: argparse.Namespace, output_path: str) -> str:
    """
    (修改版) 对 embeddings 应用 PCA 并保存。
    如果 pca_dim > 0，则同时保存原始文件和 PCA 降维后的 .npy 文件。
    不再保存 .pca 和 .scaler 模型文件。
    返回最终使用的文件路径（PCA 后的路径，如果执行了 PCA）。
    """
    pca_dim = getattr(args, 'pca_dim', 0) 

    if not isinstance(original_embeddings, np.ndarray) or original_embeddings.size == 0:
         print("[ERROR] 输入的 embeddings 无效，无法进行 PCA 或保存。")
         return "" 

    # --- 情况 1 & 2: 不执行 PCA 或无法执行 ---
    if pca_dim <= 0 or original_embeddings.shape[1] <= pca_dim:
        if pca_dim <= 0:
            print("pca_dim <= 0，跳过 PCA。")
        else:
            print(f"原始维度 ({original_embeddings.shape[1]}) <= 目标维度 ({pca_dim})，跳过 PCA。")
        
        try:
            np.save(output_path, original_embeddings)
            print(f"✅ 原始嵌入已保存至: {output_path} (维度: {original_embeddings.shape})")
            return output_path # 返回原始路径
        except Exception as e:
            print(f"[ERROR] 保存原始嵌入失败 ({output_path}): {e}")
            return "" 

    # --- 情况 3: 执行 PCA ---
    print(f"\n应用 PCA 降维，目标维度: {pca_dim}")
    try:
        # ✅ 第一步：先保存原始文件
        np.save(output_path, original_embeddings)
        print(f"✅ 原始嵌入已保存至: {output_path} (维度: {original_embeddings.shape})")

        # ✅ 第二步：执行 PCA
        pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=42)
        print("  -> 标准化数据 (StandardScaler)...")
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(original_embeddings)
        
        print(f"  -> 执行 PCA (n_components={pca_dim})...")
        reduced_emb = pca.fit_transform(scaled_embeddings).astype(np.float32)
        kept_variance = float(np.sum(pca.explained_variance_ratio_))

        # ✅ 第三步：构建 PCA 向量文件路径
        output_path_obj = Path(output_path)
        base = output_path_obj.with_suffix("") # 去掉 .npy
        pca_vec_path_str = f"{base}-pca{pca_dim}.npy"
        
        # 🚨 (移除) pca_model_path_str = f"{base}-pca{pca_dim}.pca"
        # 🚨 (移除) scaler_model_path_str = f"{base}-pca{pca_dim}.scaler"

        # ✅ 第四步：保存 PCA 向量文件
        np.save(pca_vec_path_str, reduced_emb)
        
        # 🚨 (移除) joblib.dump(pca, pca_model_path_str)
        # 🚨 (移除) joblib.dump(scaler, scaler_model_path_str) 

        print(f"✅ PCA 向量已保存: {pca_vec_path_str}  形状={reduced_emb.shape}  保留方差={kept_variance:.4f}")
        # 🚨 (移除) print(f"✅ PCA 模型已保存: ...")
        # 🚨 (移除) print(f"✅ Scaler 模型已保存: ...")
        
        # ✅ 第五步：返回 PCA 降维后的文件路径
        return pca_vec_path_str 

    except Exception as e:
        print(f"[ERROR] PCA 失败: {e}")
        print(f"PCA 失败，但原始嵌入可能已保存至: {output_path}")
        return output_path if os.path.exists(output_path) else ""