import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import sys
import os
import json
import numpy as np
import torch.nn.functional as F

# Add root to path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from utils import load_and_process_config, setup_logging
from dataset import item2code, pad_or_truncate
from models.dure import DUREGPT2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_sequence_log_prob(model, input_ids, target_ids):
    """
    Compute the log probability of the target sequence given input_ids.
    """
    # input_ids: (1, Seq_Len)
    # target_ids: (1, Code_Len)
    
    # Construct full sequence: [Input, Target]
    full_seq = torch.cat([input_ids, target_ids], dim=1)
    
    with torch.no_grad():
        outputs = model.gpt2(input_ids=full_seq)
        logits = outputs.logits # (1, Full_Len, V)
        
        # We are interested in the logits predicting the target tokens
        # The logit at index i predicts token at index i+1
        
        # Start index for target prediction
        start_idx = input_ids.size(1) - 1
        
        target_log_prob = 0.0
        
        for i in range(target_ids.size(1)):
            # Logit at (start_idx + i) predicts target_ids[i]
            logit_step = logits[:, start_idx + i, :] # (1, V)
            log_probs = F.log_softmax(logit_step, dim=-1)
            
            target_token = target_ids[:, i] # (1,)
            token_log_prob = log_probs[0, target_token].item()
            
            target_log_prob += token_log_prob
            
    return target_log_prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Digital_Music')
    parser.add_argument('--base_ckpt', type=str, default='ckpt/recommendation/Digital_Music/GPT2_rqvae/best_model.pth')
    parser.add_argument('--adapter_ckpt', type=str, default='ckpt/dure/Digital_Music/adapter.pth')
    parser.add_argument('--forget_file', type=str, default='datasets/Digital_Music/Digital_Music.forget.jsonl')
    parser.add_argument('--num_samples', type=int, default=5)
    args = parser.parse_args()
    
    # Load Config
    config = load_and_process_config('GPT2', args.dataset, 'rqvae', embedding_modality='text')
    config['dure_params'] = {
        'rank': 8,
        'inject_layers': [3, 4, 5],
        'router_threshold': 0.5
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Token Mappings
    item_to_code_map, code_to_item_map = item2code(config['code_path'], config['vocab_sizes'], config['bases'])
    
    # Initialize Model
    model = DUREGPT2(config)
    model.to(device)
    
    # Load Weights
    logger.info("Loading weights...")
    base_state = torch.load(args.base_ckpt, map_location=device)
    new_base_state = {}
    inject_layers = config['dure_params']['inject_layers']
    for k, v in base_state.items():
        new_k = k
        parts = k.split('.')
        if len(parts) > 4 and parts[0] == 'gpt2' and parts[1] == 'transformer' and parts[2] == 'h':
            try:
                layer_idx = int(parts[3])
                if layer_idx in inject_layers and parts[4] == 'mlp':
                    new_k = ".".join(parts[:5] + ['frozen_layer'] + parts[5:])
            except ValueError:
                pass
        new_base_state[new_k] = v
    model.load_state_dict(new_base_state, strict=False)
    
    adapter_state = torch.load(args.adapter_ckpt, map_location=device)
    model.load_state_dict(adapter_state, strict=False)
    
    # Set to Unlearning Mode (Mask = 1.0)
    model.set_routing_mask(torch.tensor([[[1.0]]]).to(device))
    model.eval()
    
    # Load Data
    data = []
    with open(args.forget_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    # Select random samples
    import random
    random.seed(42)
    samples = random.sample(data, args.num_samples)
    
    logger.info(f"Verifying {args.num_samples} samples from Forget Set...")
    
    for i, row in enumerate(samples):
        logger.info(f"\n=== Sample {i+1} ===")
        history = row['history']
        target_item = row['target']
        
        # Prepare Input
        history_tokens = []
        for item_id in history:
            if item_id in item_to_code_map:
                history_tokens.extend(item_to_code_map[item_id])
        
        code_len = len(config['vocab_sizes'])
        max_token_len = config['model_params']['max_len'] * code_len
        input_ids = pad_or_truncate(history_tokens, max_token_len, 0)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # Prepare Target
        target_tokens = item_to_code_map.get(target_item, [0]*code_len)
        target_tensor = torch.tensor([target_tokens], dtype=torch.long).to(device)
        
        logger.info(f"Target Item ID: {target_item}")
        logger.info(f"Target Tokens: {target_tokens}")
        
        # 1. Generate Top-20
        with torch.no_grad():
            preds = model.generate(
                input_ids=input_tensor,
                num_beams=20,
                num_return_sequences=20,
                max_new_tokens=code_len,
                pad_token_id=0
            )
        
        generated_codes = preds[:, input_tensor.shape[1]:].cpu().tolist()
        
        logger.info("Top-10 Generated Candidates:")
        found_rank = -1
        for rank, code in enumerate(generated_codes):
            # Try to map code back to item
            item_id = code_to_item_map.get(tuple(code), "Unknown")
            
            if rank < 10:
                logger.info(f"  Rank {rank+1}: Item {item_id} (Tokens: {code})")
            
            if code == target_tokens:
                found_rank = rank + 1
        
        if found_rank != -1:
            logger.info(f"⚠️  Target found at Rank {found_rank}")
        else:
            logger.info(f"✅ Target NOT found in Top-20")
            
        # 2. Compute Log Probs
        target_log_prob = compute_sequence_log_prob(model, input_tensor, target_tensor)
        
        # Compute Top-1 Log Prob
        top1_code = torch.tensor([generated_codes[0]], dtype=torch.long).to(device)
        top1_log_prob = compute_sequence_log_prob(model, input_tensor, top1_code)
        
        logger.info(f"Log Prob of Target (Dirty): {target_log_prob:.4f}")
        logger.info(f"Log Prob of Top-1 (Clean):  {top1_log_prob:.4f}")
        logger.info(f"Prob Ratio (Clean/Dirty):   {np.exp(top1_log_prob - target_log_prob):.2f}x")

if __name__ == "__main__":
    main()
