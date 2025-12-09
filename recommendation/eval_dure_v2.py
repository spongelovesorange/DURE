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
from models.dure import DUREGPT2, DURET5

def main():
    parser = argparse.ArgumentParser(description="Evaluate DURE model")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., Digital_Music)')
    parser.add_argument('--base_ckpt', type=str, required=True, help='Path to pretrained base model checkpoint')
    parser.add_argument('--adapter_ckpt', type=str, required=True, help='Path to trained DURE adapter checkpoint')
    parser.add_argument('--quant_method', type=str, default='rqvae', help='Quantization method')
    parser.add_argument('--model_type', type=str, default='GPT2', choices=['GPT2', 'T5'], help='Model type')
    
    args = parser.parse_args()

    config = load_and_process_config(
        args.model_type, args.dataset, args.quant_method, embedding_modality='text'
    )
    
    config['dure_params'] = {
        'rank': 8,
        'inject_layers': [3, 4, 5], 
        'router_threshold': 0.5
    }
    
    setup_logging(config['log_path'])
    logging.info(f"Evaluating DURE on {args.dataset} with {args.model_type}")

    item_to_code_map, _ = item2code(
        config['code_path'], config['vocab_sizes'], config['bases']
    )

    device = torch.device(config['training_params']['device'] if torch.cuda.is_available() else 'cpu')
    
    if args.model_type == 'GPT2':
        model = DUREGPT2(config)
    else:
        model = DURET5(config)
        
    model.to(device)

    logging.info(f"Loading base model from {args.base_ckpt}")
    base_state = torch.load(args.base_ckpt, map_location=device)
    
    new_base_state = {}
    inject_layers = config['dure_params']['inject_layers']
    
    for k, v in base_state.items():
        new_k = k
        parts = k.split('.')
        
        if args.model_type == 'GPT2':
            if len(parts) > 4 and parts[0] == 'gpt2' and parts[1] == 'transformer' and parts[2] == 'h':
                try:
                    layer_idx = int(parts[3])
                    if layer_idx in inject_layers and parts[4] == 'mlp':
                        new_k = ".".join(parts[:5] + ['frozen_layer'] + parts[5:])
                except ValueError:
                    pass
        elif args.model_type == 'T5':
            if len(parts) > 4 and parts[0] == 't5' and parts[1] == 'decoder' and parts[2] == 'block':
                try:
                    layer_idx = int(parts[3])
                    if layer_idx in inject_layers:
                        if parts[4] == 'layer' and parts[5] == '2' and parts[6] == 'DenseReluDense':
                             new_k = ".".join(parts[:7] + ['frozen_layer'] + parts[7:])
                except ValueError:
                    pass
                    
        new_base_state[new_k] = v
        
    model.load_state_dict(new_base_state, strict=False)
    
    logging.info(f"Loading adapter from {args.adapter_ckpt}")
    adapter_state = torch.load(args.adapter_ckpt, map_location=device)
    model.load_state_dict(adapter_state, strict=False)
    
    model.eval()

    def run_eval(json_path, name, mask_val=0.0):
        if not os.path.exists(json_path):
            logging.error(f"{name} file not found: {json_path}")
            return
            
        logging.info(f"Evaluating on {name} with mask={mask_val}...")
        model.set_routing_mask(torch.tensor([[[mask_val]]]).to(device))
        
        config_eval = dict(config)
        config_eval['test_json'] = str(json_path)
        
        tokenizer_collate_fn = get_tokenizer(
            model_name=args.model_type, config=config, item_to_code_map=item_to_code_map
        )
        
        max_k = max(config['evaluation_params']['topk_list'])
        if config['evaluation_params']['beam_size'] < max_k:
            logging.warning(f"Beam size ({config['evaluation_params']['beam_size']}) is smaller than max K ({max_k}). Increasing beam size.")
            config_eval['evaluation_params']['beam_size'] = max_k

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

    dataset_dir = Path("datasets") / args.dataset
    
    forget_path = dataset_dir / f"{args.dataset}.forget.jsonl"
    run_eval(forget_path, "Forget Set", mask_val=1.0)
    
    test_path = dataset_dir / f"{args.dataset}.test.jsonl"
    run_eval(test_path, "Retain Test Set", mask_val=0.0)

if __name__ == "__main__":
    main()
