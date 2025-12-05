import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import sys
import os

# Add root to path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from utils import load_and_process_config, setup_logging
from dataset import GenRecDataset, item2code
from tokenizer import get_tokenizer
from trainer import evaluate
from models.decoder.GPT2 import GPT2

def main():
    parser = argparse.ArgumentParser(description="Evaluate Base GPT2 model")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., Digital_Music)')
    parser.add_argument('--base_ckpt', type=str, required=True, help='Path to pretrained base model checkpoint')
    parser.add_argument('--quant_method', type=str, default='rqvae', help='Quantization method')
    
    args = parser.parse_args()

    # 1. Load Config
    config = load_and_process_config(
        'GPT2', args.dataset, args.quant_method, embedding_modality='text'
    )
    
    setup_logging(config['log_path'])
    logging.info(f"Evaluating Base GPT2 on {args.dataset}")

    # 2. Prepare Token Mappings
    item_to_code_map, _ = item2code(
        config['code_path'], config['vocab_sizes'], config['bases']
    )

    # 3. Initialize Model
    device = torch.device(config['training_params']['device'] if torch.cuda.is_available() else 'cpu')
    model = GPT2(config)
    model.to(device)

    # 4. Load Weights
    logging.info(f"Loading base model from {args.base_ckpt}")
    
    # 检查路径是否是相对路径，如果是，尝试解析为绝对路径
    ckpt_path = Path(args.base_ckpt)
    if not ckpt_path.is_absolute():
        # 尝试相对于当前工作目录
        if not ckpt_path.exists():
             # 尝试相对于项目根目录
             repo_root = Path(__file__).resolve().parent.parent
             ckpt_path = repo_root / args.base_ckpt
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    base_state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(base_state, strict=False)
    model.eval()

    # 5. Evaluation Function
    def run_eval(json_path, name):
        if not os.path.exists(json_path):
            logging.error(f"{name} file not found: {json_path}")
            return
            
        logging.info(f"Evaluating on {name}...")
        
        # Override test_json in config
        config_eval = dict(config)
        config_eval['test_json'] = str(json_path)
        
        # Build Dataset & Loader
        tokenizer_collate_fn = get_tokenizer(
            model_name='GPT2', config=config, item_to_code_map=item_to_code_map
        )
        
        dataset = GenRecDataset(config=config_eval, mode='test')
        loader = DataLoader(
            dataset,
            batch_size=config['evaluation_params']['batch_size'],
            shuffle=False,
            num_workers=4,
            collate_fn=tokenizer_collate_fn
        )
        
        results = evaluate(
            model,
            loader,
            config['evaluation_params']['topk_list'],
            device
        )
        logging.info(f"Results for {name}: {results}")
        
        print(f"\n{'='*10} {name} Results {'='*10}")
        print(f"{'Metric':<12} | {'Value':<10}")
        print("-" * 25)
        
        def sort_key(k):
            try:
                name, val = k.split('@')
                return (name, int(val))
            except:
                return (k, 0)
            
        for k in sorted(results.keys(), key=sort_key):
            print(f"{k:<12} | {results[k]:.4f}")
        print("=" * 25 + "\n")
        
        return results

    # 6. Run Evaluations
    dataset_dir = Path("datasets") / args.dataset
    
    # 6.1 Forget Set (Check if base model knows them)
    forget_path = dataset_dir / f"{args.dataset}.forget.jsonl"
    run_eval(forget_path, "Forget Set (Base Model)")
    
    # 6.2 Retain Test Set
    test_path = dataset_dir / f"{args.dataset}.test.jsonl"
    run_eval(test_path, "Retain Test Set (Base Model)")

if __name__ == "__main__":
    main()
