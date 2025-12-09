import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import zlib

# Add root to path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from models.dure import DUREGPT2, DURET5
from dure_dataset import DUREDataset, collate_fn_dure
from dure_loss import hierarchical_dpo_loss, EmbeddingContrastiveLoss, DirtyOrthogonalityLoss
from dataset import item2code, process_jsonl
from models.decoder.GPT2 import GPT2
from models.decoder.T5 import T5
from utils import load_and_process_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_pseudo_labels(model, input_ids, code_len, vocab_sizes, bases):
    """
    Generate Top-1 prediction (Semantic ID) using the frozen model.
    """
    model.eval()
    
    if hasattr(model, 't5'):
        # T5 Generation
        batch_size = input_ids.size(0)
        decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device) # Start token (0?)
        
        generated_tokens = []
        
        with torch.no_grad():
            for k in range(code_len):
                outputs = model.t5(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits # (B, Dec_Seq, V)
                next_token_logits = logits[:, -1, :]
                
                start = bases[k] + 1
                end = bases[k] + vocab_sizes[k] + 1
                
                mask = torch.full_like(next_token_logits, float('-inf'))
                mask[:, start:end] = 0
                masked_logits = next_token_logits + mask
                
                next_token = torch.argmax(masked_logits, dim=-1).unsqueeze(1)
                generated_tokens.append(next_token)
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
                
        return torch.cat(generated_tokens, dim=1)
        
    else:
        # GPT2 Generation
        batch_size = input_ids.size(0)
        curr_ids = input_ids.clone()
        generated_tokens = []
        
        with torch.no_grad():
            for k in range(code_len):
                outputs = model.gpt2(input_ids=curr_ids)
                logits = outputs.logits 
                next_token_logits = logits[:, -1, :]
                
                start = bases[k] + 1
                end = bases[k] + vocab_sizes[k] + 1
                
                mask = torch.full_like(next_token_logits, float('-inf'))
                mask[:, start:end] = 0
                masked_logits = next_token_logits + mask
                
                next_token = torch.argmax(masked_logits, dim=-1).unsqueeze(1)
                generated_tokens.append(next_token)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
                
        return torch.cat(generated_tokens, dim=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--data_root', type=str, default='/data/GenRec-Factory-QIU/datasets')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--model_type', type=str, default='GPT2', choices=['GPT2', 'T5'], help='Model type')
    parser.add_argument('--sharding_rate', type=float, default=0.5, help='Gradient sharding rate')
    parser.add_argument('--lambda_ecl', type=float, default=0.1, help='Weight for ECL loss')
    parser.add_argument('--lambda_dol', type=float, default=0.1, help='Weight for DOL loss')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    args = parser.parse_args()

    config = load_and_process_config(
        args.model_type, args.dataset, 'rqvae', embedding_modality='text'
    )
    
    config['dure_params'] = {
        'rank': 8,
        'inject_layers': [3, 4, 5] if args.model_type == 'GPT2' else [3, 4, 5], 
        'gamma_weights': [1.0, 1.0, 1.0],
        'beta': 0.1,
        'sharding_rate': args.sharding_rate
    }
    config['training_params']['epochs'] = args.epochs
    lambda_ecl = args.lambda_ecl
    lambda_dol = args.lambda_dol
    
    dataset_dir = Path(args.data_root) / args.dataset
    config['data_paths'] = {}
    config['data_paths']['train_data'] = str(dataset_dir / f"{args.dataset}.forget.jsonl")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"ckpt/dure/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config['output_dir'] = str(output_dir)
    
    device = config['training_params']['device']
    vocab_sizes = config['vocab_sizes']
    bases = config['bases']
    
    item_to_code, _ = item2code(config['code_path'], vocab_sizes, bases)
    
    logger.info(f"Initializing Base {args.model_type}...")
    if args.model_type == 'GPT2':
        base_model = GPT2(config)
        ModelClass = DUREGPT2
    else:
        base_model = T5(config)
        ModelClass = DURET5
        
    pretrained_path = f"ckpt/recommendation/{args.dataset}/{args.model_type}_rqvae/best_model.pth"
    
    if os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained model from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        base_model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning(f"Pretrained model not found at {pretrained_path}. Using random initialization.")

    base_model.to(device)
    
    logger.info("Initializing DURE Model...")
    dure_model = ModelClass(config)
    
    logger.info("Loading and adjusting base model weights...")
    if os.path.exists(pretrained_path):
        base_state = state_dict 
        new_state_dict = {}
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
                        
            new_state_dict[new_k] = v

        dure_model.load_state_dict(new_state_dict, strict=False)
    
    dure_model.to(device)
    
    logger.info("Loading Data...")
    raw_data = []
    with open(config['data_paths']['train_data'], 'r') as f:
        for i, line in enumerate(f):
            raw_data.append(json.loads(line))
    
    raw_data.sort(key=lambda x: x['user'])
            
    dataset = DUREDataset(raw_data, item_to_code, config['model_params']['max_len'], vocab_sizes, bases)
    gen_batch_size = 128
    gen_dataloader = DataLoader(dataset, batch_size=gen_batch_size, collate_fn=collate_fn_dure, shuffle=False)
    
    pseudo_labels_path = Path(config['output_dir']) / "pseudo_labels.pt"
    if os.path.exists(pseudo_labels_path):
        logger.info(f"Loading precomputed pseudo labels from {pseudo_labels_path}")
        all_labels_win = torch.load(pseudo_labels_path)
    else:
        logger.info("Precomputing pseudo labels (labels_win)...")
        all_labels_win = []
        base_model.eval()
        with torch.no_grad():
            for batch in tqdm(gen_dataloader, desc="Generating Pseudo Labels"):
                input_ids = batch['input_ids'].to(device)
                labels_win = generate_pseudo_labels(base_model, input_ids, config['code_len'], vocab_sizes, bases)
                all_labels_win.append(labels_win.cpu())
        
        all_labels_win = torch.cat(all_labels_win, dim=0)
        torch.save(all_labels_win, pseudo_labels_path)
        logger.info(f"Saved pseudo labels to {pseudo_labels_path}")

    dataloader = DataLoader(dataset, batch_size=config['training_params']['batch_size'], collate_fn=collate_fn_dure, shuffle=False)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, dure_model.parameters()), lr=float(config['training_params']['lr']))
    
    logger.info("Starting DURE Training (User-Level Sharding)...")
    dure_model.train()
    
    ecl_criterion = EmbeddingContrastiveLoss(margin=1.0).to(device)
    dol_criterion = DirtyOrthogonalityLoss().to(device)
    
    for epoch in range(config['training_params']['epochs']):
        total_loss = 0
        
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(device)
            labels_lose = batch['labels_lose'].to(device)
            user_ids = batch['user_ids']
            current_user = user_ids[0]
            
            start_idx = i * config['training_params']['batch_size']
            end_idx = start_idx + input_ids.size(0)
            labels_win = all_labels_win[start_idx:end_idx].to(device)
            
            dure_model.set_routing_mask(torch.ones(input_ids.size(0), 1, 1).to(device))
            
            if args.model_type == 'GPT2':
                input_win = torch.cat([input_ids, labels_win], dim=1)
                input_lose = torch.cat([input_ids, labels_lose], dim=1)
                
                out_win = dure_model.gpt2(input_ids=input_win, output_hidden_states=True)
                logits_win = out_win.logits[:, -config['code_len']-1:-1, :]
                
                out_lose = dure_model.gpt2(input_ids=input_lose)
                logits_lose = out_lose.logits[:, -config['code_len']-1:-1, :]
                
                last_hidden = out_win.hidden_states[-1]
                seq_len_input = input_ids.size(1)
                anchor_state = last_hidden[:, seq_len_input-1, :]
                
                embedding_layer = dure_model.gpt2.transformer.wte
                
                with torch.no_grad():
                    ref_out_win = base_model.gpt2(input_ids=input_win)
                    ref_logits_win = ref_out_win.logits[:, -config['code_len']-1:-1, :]
                    
                    ref_out_lose = base_model.gpt2(input_ids=input_lose)
                    ref_logits_lose = ref_out_lose.logits[:, -config['code_len']-1:-1, :]
                    
            else: # T5
                # T5 Forward: input_ids (Encoder), labels (Decoder)
                out_win = dure_model.t5(input_ids=input_ids, labels=labels_win, output_hidden_states=True)
                logits_win = out_win.logits
                
                out_lose = dure_model.t5(input_ids=input_ids, labels=labels_lose)
                logits_lose = out_lose.logits
                
                # Anchor State: T5 Encoder last hidden state? Or Decoder?
                # DURE paper says "User Representation".
                # In T5, we can use Encoder's last token or mean.
                # out_win.encoder_last_hidden_state (B, Seq, Hidden)
                encoder_hidden = out_win.encoder_last_hidden_state
                # Take mean or last
                anchor_state = encoder_hidden.mean(dim=1)
                
                embedding_layer = dure_model.t5.shared
                
                with torch.no_grad():
                    ref_out_win = base_model.t5(input_ids=input_ids, labels=labels_win)
                    ref_logits_win = ref_out_win.logits
                    
                    ref_out_lose = base_model.t5(input_ids=input_ids, labels=labels_lose)
                    ref_logits_lose = ref_out_lose.logits

            code_level_1_win = labels_win[:, 0]
            code_level_1_lose = labels_lose[:, 0]
            
            emb_clean = embedding_layer(code_level_1_win)
            emb_dirty = embedding_layer(code_level_1_lose)
            
            loss_ecl = ecl_criterion(anchor_state, emb_clean, emb_dirty)
            loss_dol = dol_criterion(anchor_state, emb_dirty)
            
            loss_dpo = hierarchical_dpo_loss(
                logits_win, ref_logits_win, labels_win, labels_lose, 
                config['dure_params']['gamma_weights'], config['dure_params']['beta']
            )
            
            loss = loss_dpo + lambda_ecl * loss_ecl + lambda_dol * loss_dol
            
            optimizer.zero_grad()
            loss.backward()
            
            sharding_rate = config['dure_params'].get('sharding_rate', 0.5)
            user_seed = zlib.adler32(current_user.encode('utf-8'))
            rng_state = torch.get_rng_state()
            torch.manual_seed(user_seed)
            
            for name, param in dure_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if 'lora_' in name:
                        mask = torch.bernoulli(torch.full_like(param.grad, 1 - sharding_rate)).to(device)
                        param.grad *= mask
            
            torch.set_rng_state(rng_state)
            optimizer.step()
            total_loss += loss.item()

    logger.info("Saving Side Memory...")
    adapter_state = {k: v for k, v in dure_model.state_dict().items() if 'adapters' in k or 'router' in k}
    save_path = Path(config['output_dir']) / "adapter.pth"
    torch.save(adapter_state, save_path)
    logger.info(f"Training Complete. Adapter saved to {save_path}")

if __name__ == "__main__":
    main()
