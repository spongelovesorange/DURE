import argparse
import logging
import torch
import torch.optim as optim
import os
import pprint
from typing import Optional # ✅ (新增) 导入 Optional

# ✅ 1. 從 torch.utils.data 直接導入 DataLoader
from torch.utils.data import DataLoader 
from dataset import GenRecDataset, item2code
from tokenizer import get_tokenizer       
from trainer import train_one_epoch, evaluate
from utils import load_and_process_config, setup_logging, set_seed, get_model_class

import sys
from pathlib import Path

# 获取项目根目录路径（main.py 的上一级目录的上一级）
root = Path(__file__).resolve().parent  # recommendation/
root_parent = root.parent               # 项目根目录

if str(root_parent) not in sys.path:
    sys.path.insert(0, str(root_parent))

from recommendation.models.generation.prefix_tree import Trie, build_trie_from_codebook


def main():
    # === 1. 解析命令列參數 ===
    parser = argparse.ArgumentParser(description="GenRec Universal Training Pipeline")
    parser.add_argument('--model', type=str, required=True, help='模型名稱 (e.g., TIGER, GPT2, RPG)')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名稱 (e.g., Beauty)')
    parser.add_argument('--quant_method', type=str, required=True, choices=['rkmeans', 'rvq', 'rqvae', 'opq', 'pq', 'vqvae'], help='量化方法')
    parser.add_argument('--embedding_modality', type=str, default='text', choices=['text'], help='量化模态类型，对应不同的 codebook (默认 text)')

    
    # ✅ (已移除) 删除了 --no_trie 命令行参数
    
    args = parser.parse_args()

    # === 2. 載入並處理設定檔 ===
    config = load_and_process_config(
        args.model, 
        args.dataset, 
        args.quant_method,
        embedding_modality=args.embedding_modality
    )


    # === 3. 初始化 (日誌, 隨機種子) ===
    setup_logging(config['log_path'])
    set_seed(config['training_params']['seed'])
    logging.info(f"Configuration loaded for {args.model} on {args.dataset} with {args.quant_method}.")
    logging.info("=" * 50)
    config_str = pprint.pformat(config)
    logging.info("\n" + config_str)
    logging.info("=" * 50)

    # === 4. 設定設備 ===
    device = torch.device(config['training_params']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    num_workers = config['training_params'].get('num_workers', 4)
    # 性能微调：开启 cudnn.benchmark 和 TF32（若可用）
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision('high')  # 允许 TF32 提速（A100/L20等）
        except Exception:
            pass
        # 优先启用 Flash/Mem-Efficient SDPA 注意不同 PyTorch 版本API
        try:
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
        except Exception:
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(False)
            except Exception:
                pass
    
    # === 5. ✅ (顺序调整) 载入 item_to_code 映射 ===
    # (必须在创建模型之前完成，因为Trie依赖它)
    logging.info("Loading item to code mapping...")
    item_to_code_map, _ = item2code(
        config['code_path'],
        config['vocab_sizes'],
        config['bases']
    )
    logging.info(f"Item to code map loaded. Total items mapped: {len(item_to_code_map)}")

    # === 6. ✅ (修改) 根据 config 构建前缀树 ===
    prefix_trie: Optional[Trie] = None
    
    # 检查 config['evaluation_params'] 中的 'use_prefix_trie' 标志
    # 默认值为 False (如果您希望默认不使用)
    use_trie = config.get('evaluation_params', {}).get('use_prefix_trie', False) 
    
    if use_trie and build_trie_from_codebook is not None:
        logging.info("Building Prefix Trie (enabled in config)...")
        
        # 获取所有合法的 code token 序列
        all_token_sequences = list(item_to_code_map.values())
        
        # 获取 EOS token ID
        eos_token_id = config['token_params']['eos_token_id']
        
        prefix_trie = build_trie_from_codebook(
            token_sequences=all_token_sequences,
            eos_token_id=eos_token_id
        )
    elif use_trie:
        logging.warning("Config requested Prefix Trie, but 'utils.prefix_trie' module was not found.")
    else:
        logging.info("Prefix Trie is DISABLED (default or as per config).")


    # === 7. ✅ (修改) 创建模型 (将Trie注入) ===
    logging.info(f"Dynamically loading model: {args.model}")
    ModelClass = get_model_class(args.model)
    
    # ✅ (修改) 将 config 和 prefix_trie (可能是 None) 传递给模型
    #    (我们假设 ModelClass 的 __init__ 接受 prefix_trie=None)
    model = ModelClass(config, prefix_trie=prefix_trie) 
    
    model.to(device)
    logging.info(model.n_parameters)
    logging.info("=" * 50)
    # 使用 fused AdamW（支持 CUDA 的 PyTorch 会更快），失败则回退到常规 AdamW/Adam
    lr = float(config['training_params']['lr'])
    optimizer = None
    try:
        optimizer = optim.AdamW(model.parameters(), lr=lr, fused=True)  # PyTorch 2.0+
        logging.info("Using fused AdamW optimizer")
    except Exception:
        try:
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            logging.info("Using AdamW optimizer")
        except Exception:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            logging.info("Using Adam optimizer (fallback)")

    # 学习率 warmup：减少大 batch + AMP 初期不稳定
    warmup_steps = int(config['training_params'].get('warmup_steps', 0))
    scheduler = None
    if warmup_steps and warmup_steps > 0:
        try:
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
            logging.info(f"Using LinearLR warmup: start_factor=0.1, total_iters={warmup_steps}")
        except Exception as e:
            logging.warning(f"Failed to create LinearLR warmup scheduler: {e}")

    # === 8. (顺序调整) 初始化模型专属的 Tokenizer ===
    logging.info(f"Initializing tokenizer for model: {args.model}")
    tokenizer_collate_fn = get_tokenizer(
        model_name=args.model,
        config=config,
        item_to_code_map=item_to_code_map
    )
    logging.info("Tokenizer initialized.")

    # === 9. (顺序调整) 創建數據集與 DataLoader ===
    logging.info("Creating Datasets...")
    train_dataset = GenRecDataset(config=config, mode='train')
    validation_dataset = GenRecDataset(config=config, mode='valid')
    test_dataset = GenRecDataset(config=config, mode='test')

    logging.info("Creating DataLoaders...")
    
    is_gpu_training = (torch.cuda.is_available() and num_workers > 0)
    prefetch_factor = config['training_params'].get('prefetch_factor', 4)
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": tokenizer_collate_fn, # 傳入 tokenizer
        "pin_memory": is_gpu_training,
        "persistent_workers": is_gpu_training if num_workers > 0 else False
    }
    if num_workers and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training_params']['batch_size'],
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config['evaluation_params']['batch_size'],
        shuffle=False, 
        **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation_params']['batch_size'],
        shuffle=False, 
        **loader_kwargs
    )

    # === 10. (顺序调整) 训练-评估循环 (已修改) ===
    best_ndcg = 0.0
    early_stop_counter = 0
    best_epoch = 0
    best_val_results = None
    best_test_results = None
    
    eval_interval = config['training_params'].get('eval_interval', 1) 
    logging.info(f"Evaluation interval set to: {eval_interval} epoch(s)")

    for epoch in range(config['training_params']['num_epochs']):
        epoch_num = epoch + 1 
        logging.info(f"--- Epoch {epoch_num}/{config['training_params']['num_epochs']} ---")
        
        # --- 训练 ---
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scheduler)
        logging.info(f"Training loss: {train_loss:.4f}")

        # --- 评估 (根据 eval_interval) ---
        if epoch_num % eval_interval == 0:
            logging.info(f"--- Evaluating at Epoch {epoch_num} ---")
            val_results = evaluate(
                model,
                validation_loader,
                config['evaluation_params']['topk_list'],
                device
            )
            logging.info(f"Validation Results: {val_results}")

            current_ndcg = val_results.get('NDCG@10', val_results.get('NDCG@20', 0.0))

            # --- 检查性能提升和 Early Stopping ---
            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                early_stop_counter = 0 
                logging.info(f"🚀 New best NDCG on validation: {best_ndcg:.4f} at Epoch {epoch_num}")

                test_results = evaluate(
                    model,
                    test_loader,
                    config['evaluation_params']['topk_list'],
                    device
                )
                logging.info(f"Test Results: {test_results}")

                best_epoch = epoch_num
                best_val_results = val_results
                best_test_results = test_results

                torch.save(model.state_dict(), config['save_path'])
                logging.info(f"Best model saved to {config['save_path']}")
            
            else:
                early_stop_counter += eval_interval 
                logging.info(f"No improvement since Epoch {best_epoch}. Early stop counter: {early_stop_counter}/{config['training_params']['early_stop'] * eval_interval}")
                if early_stop_counter >= config['training_params']['early_stop'] * eval_interval:
                    logging.info("Early stopping triggered.")
                    break
        else:
             logging.info(f"Skipping evaluation for Epoch {epoch_num}.")

    # === 11. (顺序调整) 訓練結束總結 ===
    logging.info("="*50)
    logging.info("🏁 Training Finished!")
    if best_test_results:
        logging.info(f"🏆 Best performance found at Epoch {best_epoch}")
        logging.info(f"  - Best Validation Results: {best_val_results}")
        logging.info(f"  - Corresponding Test Results: {best_test_results}")
    else:
        logging.info("No improvement was observed.")
        # Save the last model if no best model was saved (e.g. if eval_interval > num_epochs)
        # Force save/overwrite to ensure we have the latest model
        logging.info(f"Saving last model to {config['save_path']} (fallback/overwrite).")
        torch.save(model.state_dict(), config['save_path'])
    logging.info("="*50)


if __name__ == "__main__":
    main()