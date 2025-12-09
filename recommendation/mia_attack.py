import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import json
from tqdm import tqdm

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

class MIADataset(torch.utils.data.Dataset):
    def __init__(self, data_path, item_to_code, max_len, vocab_sizes, bases, pad_token=0):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.item_to_code = item_to_code
        self.max_len = max_len
        self.vocab_sizes = vocab_sizes
        self.bases = bases
        self.pad_token = pad_token
        self.code_len = len(vocab_sizes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        history = row['history']
        target = row['target']

        # Flatten history
        history_tokens = []
        for item_id in history:
            if item_id in self.item_to_code:
                history_tokens.extend(self.item_to_code[item_id])
        
        # Target tokens
        if target in self.item_to_code:
            target_tokens = self.item_to_code[target]
        else:
            target_tokens = [0] * self.code_len # Should not happen

        # Construct Input: [History, Target]
        # We want to predict Target given History.
        # GPT2 is causal, so we feed [History, Target] and look at loss on Target positions.
        
        # Truncate history if needed
        # Total length limit? Let's say max_len items for history
        max_hist_tokens = self.max_len * self.code_len
        if len(history_tokens) > max_hist_tokens:
            history_tokens = history_tokens[-max_hist_tokens:]
            
        input_ids = history_tokens + target_tokens
        
        # Create mask: 0 for history, 1 for target
        # We only want to evaluate loss on the target tokens
        loss_mask = [0] * len(history_tokens) + [1] * len(target_tokens)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'loss_mask': torch.tensor(loss_mask, dtype=torch.float),
            'target_len': len(target_tokens)
        }

def collate_fn_mia(batch):
    # Pad input_ids and loss_mask
    max_len = max([item['input_ids'].size(0) for item in batch])
    
    padded_input_ids = []
    padded_loss_masks = []
    
    for item in batch:
        input_ids = item['input_ids']
        loss_mask = item['loss_mask']
        
        padding_len = max_len - input_ids.size(0)
        
        # Pad left or right? GPT2 usually left padding for generation, but for training/loss right padding is fine if masked correctly.
        # Let's use left padding to align the end (target) if we were generating, but here we just compute loss.
        # Actually, for causal LM, right padding is standard for training batches, with attention mask.
        # But here we are just computing loss, let's do right padding with 0.
        
        padded_ids = torch.cat([input_ids, torch.zeros(padding_len, dtype=torch.long)])
        padded_mask = torch.cat([loss_mask, torch.zeros(padding_len, dtype=torch.float)])
        
        padded_input_ids.append(padded_ids)
        padded_loss_masks.append(padded_mask)
        
    return {
        'input_ids': torch.stack(padded_input_ids),
        'loss_mask': torch.stack(padded_loss_masks)
    }

def compute_losses(model, dataloader, device):
    model.eval()
    all_losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Losses"):
            input_ids = batch['input_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            
            # Forward pass
            # GPT2 outputs logits for the next token
            outputs = model.gpt2(input_ids=input_ids)
            logits = outputs.logits # (B, L, V)
            
            # Shift for loss calculation
            # Logits at t predict token at t+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = loss_mask[..., 1:].contiguous()
            
            # Compute Cross Entropy
            # Flatten
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Reshape back to (B, L-1)
            loss = loss.view(shift_labels.size())
            
            # Apply mask (only sum loss over target tokens)
            masked_loss = loss * shift_mask
            
            # Sum over sequence and divide by number of target tokens
            # Note: shift_mask contains 1s only for target tokens
            seq_loss = masked_loss.sum(dim=1) / (shift_mask.sum(dim=1) + 1e-9)
            
            all_losses.extend(seq_loss.cpu().numpy().tolist())
            
    return np.array(all_losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--base_ckpt', type=str, required=True)
    parser.add_argument('--adapter_ckpt', type=str, required=True)
    parser.add_argument('--forget_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
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
    item_to_code_map, _ = item2code(config['code_path'], config['vocab_sizes'], config['bases'])
    
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
    
    # Prepare DataLoaders
    logger.info("Preparing Data...")
    forget_dataset = MIADataset(args.forget_file, item_to_code_map, config['model_params']['max_len'], config['vocab_sizes'], config['bases'])
    test_dataset = MIADataset(args.test_file, item_to_code_map, config['model_params']['max_len'], config['vocab_sizes'], config['bases'])
    
    # Balance datasets (take min length)
    min_len = min(len(forget_dataset), len(test_dataset))
    # min_len = 1000 # For speed testing
    
    indices_forget = np.random.choice(len(forget_dataset), min_len, replace=False)
    indices_test = np.random.choice(len(test_dataset), min_len, replace=False)
    
    forget_subset = torch.utils.data.Subset(forget_dataset, indices_forget)
    test_subset = torch.utils.data.Subset(test_dataset, indices_test)
    
    forget_loader = DataLoader(forget_subset, batch_size=32, collate_fn=collate_fn_mia)
    test_loader = DataLoader(test_subset, batch_size=32, collate_fn=collate_fn_mia)
    
    # Compute Losses
    logger.info("Computing losses for Forget Set (Members)...")
    forget_losses = compute_losses(model, forget_loader, device)
    
    logger.info("Computing losses for Test Set (Non-Members)...")
    test_losses = compute_losses(model, test_loader, device)
    
    # MIA Analysis
    # We want to distinguish Members (Forget) from Non-Members (Test)
    # Feature: Loss
    # Label: 1 for Member, 0 for Non-Member
    # Logic: Members usually have LOWER loss.
    # So we use -Loss as the score. Higher score (-Loss) => More likely Member.
    
    y_true = np.concatenate([np.ones(len(forget_losses)), np.zeros(len(test_losses))])
    y_scores = np.concatenate([-forget_losses, -test_losses]) # Negate loss so higher is "better" (more likely member)
    
    auc = roc_auc_score(y_true, y_scores)
    
    # Calculate Accuracy at best threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Best threshold is closest to (0,1)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (y_scores >= optimal_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    logger.info("="*30)
    logger.info("MIA Experiment Results")
    logger.info("="*30)
    logger.info(f"Forget Set Mean Loss: {forget_losses.mean():.4f}")
    logger.info(f"Test Set Mean Loss:   {test_losses.mean():.4f}")
    logger.info(f"MIA AUC:              {auc:.4f}")
    logger.info(f"MIA Accuracy:         {acc:.4f}")
    logger.info("="*30)
    logger.info("Interpretation:")
    logger.info("AUC near 0.5 means Perfect Unlearning (Indistinguishable).")
    logger.info("AUC near 1.0 means Failed Unlearning (Forget Set is easily identified).")
    logger.info("AUC < 0.5 means Forget Set is 'forgotten' even more than unseen data (Over-unlearning).")

if __name__ == "__main__":
    main()
