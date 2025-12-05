# utils.py
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import numpy as np
import random
import torch
import logging
from pathlib import Path
import importlib
from collections.abc import Mapping

VALID_QUANT_METHODS = {"rkmeans", "rvq", "rqvae", "opq", "pq", 'vqvae'}

def _ensure_dir_exists(dir_path: Path):
    """确保目录存在"""
    if dir_path:
        dir_path.mkdir(parents=True, exist_ok=True)

def _recursive_update(base_dict: dict, new_dict: dict) -> dict:
    """
    遞迴地更新字典。
    如果 new_dict 中的鍵在 base_dict 中也存在且對應的值都是字典，
    則遞迴地合併它們，否則直接用 new_dict 的值覆蓋 base_dict 的值。
    """
    for key, value in new_dict.items():
        if isinstance(value, Mapping) and key in base_dict and isinstance(base_dict[key], Mapping):
            base_dict[key] = _recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def _load_quant_details(path: str, quant_method: str) -> dict:
    """
    從指定的 YAML 檔案中，根據 quant_method 載入對應的參數。
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"[Config] 根據約定，未找到量化設定檔: {path}")
    
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    if quant_method not in cfg or 'model_params' not in cfg[quant_method]:
        raise ValueError(f"[Config] 在 {path} 中缺少 '{quant_method}.model_params' 節點")
    
    mp = cfg[quant_method]['model_params']
    required_keys = ['codebook_size', 'num_levels']
    if not all(key in mp for key in required_keys):
         raise ValueError(f"[Config] 在 {path} 的 model_params 中缺少 'codebook_size' 或 'num_levels'")

    return mp


def load_and_process_config(model_name: str, dataset_name: str, quant_method: str, embedding_modality: str = 'text') -> dict:
    """
    通用配置加载器 (V6 - 支援 base.yaml 繼承與覆蓋)。
    """
    # === 1. ✅ 關鍵改動 2：依序載入 base 和 model-specific 設定檔 ===
    # 載入基礎設定檔
    # 锚定到 recommendation/ 和 repo 根，避免依赖 CWD
    rec_dir = Path(__file__).resolve().parent
    repo_root = rec_dir.parent

    base_config_path = rec_dir / "configs" / "base.yaml"
    if not base_config_path.is_file():
        raise FileNotFoundError(f"基礎設定檔未找到: {base_config_path}")
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 載入特定模型設定檔
    model_config_path = rec_dir / "configs" / f"{model_name}.yaml"
    if not model_config_path.is_file():
        raise FileNotFoundError(f"模型配置文件未找到: {model_config_path}")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
        
    # ✨ 使用遞迴更新，讓 model_config 覆蓋 base_config
    config = _recursive_update(config, model_config)

    # === 後續流程完全不變，它們現在操作的是已經合併好的 config ===
    
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config['quant_method'] = quant_method.lower()
    config['embedding_modality'] = embedding_modality.lower()
    
    if config['quant_method'] not in VALID_QUANT_METHODS:
        raise ValueError(f"不支持的量化方法: {quant_method}。可选: {VALID_QUANT_METHODS}")

    # 2. 獨立地載入量化設定檔
    quant_config_path = repo_root / "quantization" / "configs" / f"{quant_method}_config.yaml"
    quant_details = _load_quant_details(quant_config_path, config['quant_method'])
    
    # === 3. 格式化和派生「數據」與「輸出」的路徑 ===
    # 這部分的路徑模板仍然來自 TIGER.yaml，是合理的，因為它定義了數據存放格式
    # 3. 格式化路徑
    paths = config['paths']
    # ✅ 將 model_name 加入字典
    format_args = {
        'dataset_name': dataset_name, 
        'quant_method': config['quant_method'], 
        'model_name': model_name
    }
    dataset_root = Path(paths['dataset_root'].format(**format_args))
    output_root = Path(paths['output_root'].format(**format_args))

    # 统一锚定到 repo 根目录
    if not dataset_root.is_absolute():
        dataset_root = (repo_root / dataset_root).resolve()
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()

        # === 4. 自动构造 codebook 路径 ===
    # 覆盖为基于 repo-root 的 datasets 路径
    dataset_root = repo_root / "datasets" / f"{dataset_name}"
    codebook_dir = dataset_root / "codebooks"

    mod_tag = embedding_modality.lower()
    quant_tag = config['quant_method'].lower()

    # 严格匹配指定模态和量化方法
    codebook_path = codebook_dir / f"{dataset_name}.{mod_tag}.{quant_tag}.npy"

    if not codebook_path.exists():
        raise FileNotFoundError(
            f"[FATAL] 未找到指定模态 '{mod_tag}' 的 codebook 文件！\n"
            f"期望路径: {codebook_path}\n"
            f"请确认路径及文件名与保存时一致。"
        )

    config['code_path'] = str(codebook_path)
    logging.info(f"📦 [Config] 成功加载 Codebook: {config['code_path']}")

    config['log_path'] = output_root / "training.log"
    config['save_path'] = output_root / "best_model.pth"
    config['train_json'] = dataset_root / f"{dataset_name}.train.jsonl"
    config['valid_json'] = dataset_root / f"{dataset_name}.valid.jsonl"
    config['test_json'] = dataset_root / f"{dataset_name}.test.jsonl"
    _ensure_dir_exists(output_root)

    # === 4. 根據載入的量化細節，計算詞表參數 ===
    K = int(quant_details['codebook_size'])
    num_semantic_levels = int(quant_details['num_levels'])
    has_dup_layer = quant_details.get('has_dup_layer', True) 
    
    config['codebook_size'] = K
    config['num_semantic_levels'] = num_semantic_levels

    # === 5. 校验 codebook 檔案 ===
    if not Path(config['code_path']).is_file():
        raise FileNotFoundError(f"[FATAL] 未找到 codebook: {config['code_path']}")
    codes_arr = np.load(config['code_path'], allow_pickle=True)
    codes_mat = np.vstack(codes_arr) if codes_arr.dtype == object else codes_arr
    
    expected_code_len = num_semantic_levels + 1 if has_dup_layer else num_semantic_levels
    config['code_len'] = expected_code_len

    if codes_mat.ndim != 2 or codes_mat.shape[1] != expected_code_len:
        raise ValueError(f"[FATAL] Codebook {config['code_path']} 的期望形状為 (N, {expected_code_len})，實際為 {codes_mat.shape}")

    # === 6. 計算最終詞表參數 ===
    if has_dup_layer:
        dup_max = int(codes_mat[:, -1].max()) if codes_mat.size > 0 else 0
        dup_vocab_size = dup_max + 1
        config['dup_vocab_size'] = dup_vocab_size
        vocab_sizes = [K] * num_semantic_levels + [dup_vocab_size]
    else:
        vocab_sizes = [K] * num_semantic_levels
    config['vocab_sizes'] = vocab_sizes

    bases = np.cumsum([0] + vocab_sizes[:-1]).tolist()
    config['bases'] = [int(b) for b in bases]

    max_token_id = sum(config['vocab_sizes'])
    eos_id = max_token_id + 1
    
    config['token_params'] = {
        'pad_token_id': 0, 'eos_token_id': eos_id, 'vocab_size': eos_id + 5 
    }
    
    return config


# --- 日志和随机种子函数保持不变 ---
def setup_logging(log_path: Path):
    """配置日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_path), mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

def get_model_class(model_name: str):
    """
    增强版：根据模型名称字符串，动态地从 'models' 目录及其所有子目录中
    搜索并导入对应的模型类。

    Args:
        model_name (str): 模型的名称 (e.g., "TIGER")。
                          约定：文件名和类名都应与 model_name 一致 (TIGER.py, class TIGER)。
    """
    models_root_dir = os.path.join(os.path.dirname(__file__), "models")
    model_file_name = f"{model_name}.py"
    model_module_path = None
    
    # 获取项目根目录 (recommendation 的上一级)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # os.walk 会遍历指定目录下的所有文件夹和文件
    for root, dirs, files in os.walk(models_root_dir):
        if model_file_name in files:
            # 找到了文件！现在构建 Python 的导入路径
            # 例如, root = "/data/.../recommendation/models/decoder"
            # project_root = "/data/..."
            
            # 1. 获取相对于项目根目录的路径
            # "recommendation/models/decoder"
            rel_path = os.path.relpath(root, project_root)
            
            # 2. 将文件系统路径 ('/') 替换为 Python 导入路径 ('.')
            # "recommendation.models.decoder"
            base_path = rel_path.replace(os.sep, '.')
            
            # 3. 拼接成最终的模块路径
            # "recommendation.models.decoder.GPT2"
            model_module_path = f"{base_path}.{model_name}"
            break # 找到后立刻停止搜索

    # 如果遍历完都没找到文件
    if not model_module_path:
        raise ImportError(
            f"错误：无法在 '{models_root_dir}' 目录或其任何子目录中找到模型文件 '{model_file_name}'。\n"
            f"请检查你的文件结构和 --model 参数是否正确。"
        )

    try:
        # 使用动态构建的路径来导入模块
        logging.info(f"Found model module at: {model_module_path}")
        model_module = importlib.import_module(model_module_path)
        # 从模块中获取与模型同名的类
        model_class = getattr(model_module, model_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"错误：成功找到文件，但在从 '{model_module_path}' 导入类 '{model_name}' 时失败。\n"
            f"请确保你的 Python 文件内 class 的名称 ({model_name}) 与文件名和 --model 参数完全一致。\n"
            f"原始错误: {e}"
        )

def set_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)