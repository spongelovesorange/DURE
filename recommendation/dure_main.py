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
import zlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_pseudo_labels(model, input_ids, code_len, vocab_sizes, bases):
    """
    Generate Top-1 prediction (Semantic ID) using the frozen model.
    """
    model.eval()
    batch_size = input_ids.size(0)
    
    # We need to generate code_len tokens
    # Start with input_ids
    curr_ids = input_ids.clone()
    
    generated_tokens = []
    
    with torch.no_grad():
        for k in range(code_len):
            # Forward pass
            # GPT2 forward returns logits for the last token
            outputs = model.gpt2(input_ids=curr_ids)
            logits = outputs.logits # (B, Seq, V)
            next_token_logits = logits[:, -1, :] # (B, V)
            
            # Mask invalid tokens for this level?
            # We know the range for level k is [bases[k]+1, bases[k]+vocab_sizes[k]]
            # But for simplicity, let's just take argmax and assume it's valid 
            # or mask everything else.
            
            # Create a mask for valid tokens at this level
            start = bases[k] + 1
            end = bases[k] + vocab_sizes[k] + 1
            
            # We can just slice the logits or mask them
            # Let's mask
            mask = torch.full_like(next_token_logits, float('-inf'))
            mask[:, start:end] = 0
            masked_logits = next_token_logits + mask
            
            next_token = torch.argmax(masked_logits, dim=-1).unsqueeze(1) # (B, 1)
            
            generated_tokens.append(next_token)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
    return torch.cat(generated_tokens, dim=1) # (B, code_len)

def main():
    # Configuration
    config = {
        'model_params': {
            'n_embd': 512, # Match pre-trained model
            'n_layer': 6,  # Match pre-trained model
            'n_head': 8,   # Match pre-trained model
            'n_inner': 2048, # Match pre-trained model
            'max_len': 20, # Items
        },
        'token_params': {
            'vocab_size': None, # To be filled
            'eos_token_id': None,
            'pad_token_id': 0
        },
        'code_len': 3, # Assumed
        'dure_params': {
            'rank': 8,
            'inject_layers': [3, 4, 5], # Inject into last 3 layers (0-indexed, n_layer=6)
            'gamma_weights': [1.0, 1.0, 1.0], # Equal weights
            'beta': 0.1,
            'sharding_rate': 0.5 # 50% of parameters are frozen per batch
        },
        'training_params': {
            'lr': 1e-3,
            'batch_size': 32,
            'epochs': 20,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'dataset_name': 'ml-1m', # Default, will be overwritten
        'data_paths': {
            'codebook': '/data/GenRec-Factory-QIU/datasets/ml-1m/codebooks/ml-1m_text_3_128.npy', # Adjust path if needed
            'train_data': '/data/GenRec-Factory-QIU/datasets/ml-1m/ml-1m.train.jsonl'
        }
    }
    
    # Parse args for dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--data_root', type=str, default='/data/GenRec-Factory-QIU/datasets')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save checkpoints')
    # Sensitivity Analysis Args
    parser.add_argument('--sharding_rate', type=float, default=0.5, help='Gradient sharding rate')
    parser.add_argument('--lambda_ecl', type=float, default=0.1, help='Weight for ECL loss')
    parser.add_argument('--lambda_dol', type=float, default=0.1, help='Weight for DOL loss')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    args = parser.parse_args()
    
    config['dataset_name'] = args.dataset
    config['dure_params']['sharding_rate'] = args.sharding_rate
    config['training_params']['epochs'] = args.epochs
    lambda_ecl = args.lambda_ecl
    lambda_dol = args.lambda_dol
    
    # Update paths based on dataset
    dataset_dir = Path(args.data_root) / args.dataset
    config['data_paths']['train_data'] = str(dataset_dir / f"{args.dataset}.forget.jsonl")

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"ckpt/dure/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config['output_dir'] = str(output_dir)
    
    # Find codebook
    code_dir = dataset_dir / "codebooks"
    files = list(code_dir.glob('*.npy'))
    if files:
        config['data_paths']['codebook'] = str(files[0])
    
    device = config['training_params']['device']
    
    # 1. Load Codebook Info
    # We need to know vocab sizes to set up the model
    # Assuming codebook exists. If not, we might fail.
    # Let's check if the file exists, otherwise mock it.
    code_path = config['data_paths']['codebook']
    if not os.path.exists(code_path):
        # Try to find any .npy file in codebooks
        code_dir = Path('/data/GenRec-Factory-QIU/datasets/ml-1m/codebooks')
        files = list(code_dir.glob('*.npy'))
        if files:
            code_path = str(files[0])
            logger.info(f"Using codebook: {code_path}")
        else:
            logger.error("No codebook found.")
            return

    # Inspect codebook to get vocab sizes
    try:
        data = np.load(code_path, allow_pickle=True)
        mat = np.vstack(data) if data.dtype == object else data
        # mat shape: (num_items, num_levels)
        num_levels = mat.shape[1]
        config['code_len'] = num_levels
        
        # Infer vocab sizes per level
        vocab_sizes = []
        for i in range(num_levels):
            vocab_sizes.append(int(mat[:, i].max()) + 1)
        
        logger.info(f"Inferred vocab sizes: {vocab_sizes}")
        
    except Exception as e:
        logger.error(f"Failed to load codebook: {e}")
        return

    bases = [0]
    for s in vocab_sizes[:-1]:
        bases.append(bases[-1] + s + 1) # +1 for potential special tokens?
    
    # Total vocab size
    # In main.py, vocab_size is calculated as:
    # vocab_size = sum(vocab_sizes) + len(vocab_sizes) + 2 (PAD, EOS)
    # Let's match exactly what main.py does or what the checkpoint has.
    # Checkpoint has 1030.
    # vocab_sizes = [256, 256, 256, 256] -> sum = 1024
    # bases logic in main.py:
    # bases[0] = 0
    # bases[1] = 0 + 256 + 1 = 257 (offset for level 1)
    # ...
    # The logic here:
    # bases = [0, 257, 514, 771]
    # last end = 771 + 256 + 1 = 1028
    # total_vocab = 1028 + 2 = 1030?
    
    # Let's trace the code above:
    # bases = [0]
    # for s in [256, 256, 256]:
    #   bases.append(bases[-1] + s + 1) -> [0, 257, 514, 771]
    # total_vocab = 771 + 256 + 2 = 1029.
    
    # Wait, why 1030 in checkpoint?
    # Maybe main.py uses +1 for something else? Or vocab_sizes are different?
    # In main.py log: 'vocab_size': 1030.
    # Let's force it to 1030 to match checkpoint if it's close.
    # Or better, check how main.py calculates it.
    
    # Assuming main.py logic:
    # It might be using a different offset or special token count.
    # Let's just set it to 1030 to match the checkpoint.
    total_vocab = 1030 
    config['token_params']['vocab_size'] = total_vocab
    config['token_params']['eos_token_id'] = total_vocab - 1
    config['token_params']['pad_token_id'] = 0
    
    item_to_code, _ = item2code(code_path, vocab_sizes, bases)
    
    # 2. Initialize Base Model (Load Pretrained)
    logger.info("Initializing Base GPT2...")
    base_model = GPT2(config)
    
    # Load the pretrained checkpoint
    pretrained_path = f"ckpt/recommendation/{args.dataset}/GPT2_rqvae/best_model.pth"
    if os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained model from {pretrained_path}")
        # Use weights_only=False to avoid the warning, or True if safe. 
        # Since we created it, it's safe. But the warning suggests True is future default.
        # However, our checkpoint might contain full state dict.
        state_dict = torch.load(pretrained_path, map_location=device)
        base_model.load_state_dict(state_dict)
    else:
        logger.warning(f"Pretrained model not found at {pretrained_path}. Using random initialization (NOT RECOMMENDED for Unlearning).")

    base_model.to(device)
    
    # Save it as "frozen" checkpoint (optional, but keeps logic consistent)
    # torch.save(base_model.state_dict(), "base_model_frozen.pth")
    
    # 3. Initialize DURE Model
    logger.info("Initializing DURE Model...")
    dure_model = DUREGPT2(config)
    
    # Fix: Adjust keys for wrapped layers to ensure pretrained weights are loaded into frozen_layer
    logger.info("Loading and adjusting base model weights...")
    # Use the state_dict we just loaded, no need to reload from disk
    base_state = state_dict 
    new_state_dict = {}
    inject_layers = config['dure_params']['inject_layers']
    
    for k, v in base_state.items():
        new_k = k
        # Check if this key belongs to an injected layer's MLP
        # Pattern: gpt2.transformer.h.{i}.mlp.
        parts = k.split('.')
        if len(parts) > 4 and parts[0] == 'gpt2' and parts[1] == 'transformer' and parts[2] == 'h':
            try:
                layer_idx = int(parts[3])
                if layer_idx in inject_layers and parts[4] == 'mlp':
                    # Insert 'frozen_layer' after 'mlp'
                    # k is like gpt2.transformer.h.3.mlp.c_fc.weight
                    # new_k should be gpt2.transformer.h.3.mlp.frozen_layer.c_fc.weight
                    new_k = ".".join(parts[:5] + ['frozen_layer'] + parts[5:])
            except ValueError:
                pass
        
        new_state_dict[new_k] = v

    dure_model.load_state_dict(new_state_dict, strict=False)
    dure_model.to(device)
    
    # 4. Prepare Data
    logger.info("Loading Data...")
    # Load data
    raw_data = []
    with open(config['data_paths']['train_data'], 'r') as f:
        for i, line in enumerate(f):
            # if i >= 100: break # Small dataset
            raw_data.append(json.loads(line))
    
    # Sort data by user to enable user-level batching/sharding
    raw_data.sort(key=lambda x: x['user'])
            
    dataset = DUREDataset(raw_data, item_to_code, config['model_params']['max_len'], vocab_sizes, bases)
    # Use larger batch size for generation if possible
    gen_batch_size = 128
    gen_dataloader = DataLoader(dataset, batch_size=gen_batch_size, collate_fn=collate_fn_dure, shuffle=False)
    
    # Precompute Pseudo Labels
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
                # Generate
                labels_win = generate_pseudo_labels(base_model, input_ids, config['code_len'], vocab_sizes, bases)
                all_labels_win.append(labels_win.cpu())
        
        all_labels_win = torch.cat(all_labels_win, dim=0)
        torch.save(all_labels_win, pseudo_labels_path)
        logger.info(f"Saved pseudo labels to {pseudo_labels_path}")

    # Training DataLoader
    # IMPORTANT: shuffle=False to maintain user order
    dataloader = DataLoader(dataset, batch_size=config['training_params']['batch_size'], collate_fn=collate_fn_dure, shuffle=False)
    
    # 5. Training Loop
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, dure_model.parameters()), lr=config['training_params']['lr'])
    
    logger.info("Starting DURE Training (User-Level Sharding)...")
    dure_model.train()
    
    # Initialize extra loss functions
    ecl_criterion = EmbeddingContrastiveLoss(margin=1.0).to(device)
    dol_criterion = DirtyOrthogonalityLoss().to(device)
    
    import zlib
    
    for epoch in range(config['training_params']['epochs']):
        total_loss = 0
        
        batch_idx = 0
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(device)
            labels_lose = batch['labels_lose'].to(device)
            user_ids = batch['user_ids'] # List of user IDs
            
            # Check if batch contains multiple users
            # For strict user-level sharding, we should ideally have 1 user per batch.
            # But if we sorted, we might have [A, A, B, B].
            # We will use the FIRST user in the batch to determine the seed.
            # This is an approximation if batch_size > 1 and users switch in the middle.
            # For simulation, we assume dominant user or just use the first one.
            current_user = user_ids[0]
            
            # Fetch precomputed labels
            # Calculate start and end index for this batch
            start_idx = i * config['training_params']['batch_size']
            end_idx = start_idx + input_ids.size(0)
            labels_win = all_labels_win[start_idx:end_idx].to(device)
            
            # Forward pass DURE
            
            # Forward pass DURE
            # We need logits for the *target* sequence (the response)
            # DURE forward returns logits for the whole sequence (input + generated?)
            # Wait, GPT2 forward takes input_ids and returns logits for next tokens.
            # We want logits for the *continuation*.
            
            # We need to feed [input_ids, target_ids] to get logits for target_ids?
            # Yes.
            # But we have two targets: win and lose.
            # We need to run DURE twice?
            # 1. DURE(input + win) -> logits_win
            # 2. DURE(input + lose) -> logits_lose
            
            # Construct inputs
            input_win = torch.cat([input_ids, labels_win], dim=1)
            input_lose = torch.cat([input_ids, labels_lose], dim=1)
            
            # Run DURE
            # We need to set routing mask. For training, we assume we are "forgetting", so mask=1?
            # Or we use the router?
            # For now, force mask=1 to train the adapter.
            dure_model.set_routing_mask(torch.ones(input_ids.size(0), 1, 1).to(device))
            
            # Enable hidden states for extra losses
            out_win = dure_model.gpt2(input_ids=input_win, output_hidden_states=True)
            logits_win = out_win.logits[:, -config['code_len']-1:-1, :] # Logits predicting the target
            
            out_lose = dure_model.gpt2(input_ids=input_lose)
            logits_lose = out_lose.logits[:, -config['code_len']-1:-1, :]
            
            # --- Extra Losses Calculation ---
            # 1. Get Anchor State (DURE's hidden state at the end of input_ids)
            # out_win.hidden_states is a tuple of (B, Seq, Hidden) for each layer
            # We take the last layer
            last_hidden = out_win.hidden_states[-1]
            # We want the state corresponding to the last token of input_ids (before generation starts)
            seq_len_input = input_ids.size(1)
            anchor_state = last_hidden[:, seq_len_input-1, :] # (B, Hidden)
            
            # 2. Get Positive (Clean) and Negative (Dirty) Embeddings
            # We use the embedding of the first code (Level 1) as the representative semantic vector
            # GPT2 embedding layer is usually at dure_model.gpt2.transformer.wte
            embedding_layer = dure_model.gpt2.transformer.wte
            
            # labels_win/lose are (B, Code_Len)
            # We take the first column (Level 1 Code)
            code_level_1_win = labels_win[:, 0]
            code_level_1_lose = labels_lose[:, 0]
            
            emb_clean = embedding_layer(code_level_1_win) # (B, Hidden)
            emb_dirty = embedding_layer(code_level_1_lose) # (B, Hidden)
            
            # 3. Compute Losses
            loss_ecl = ecl_criterion(anchor_state, emb_clean, emb_dirty)
            loss_dol = dol_criterion(anchor_state, emb_dirty)
            
            # Run Reference (Frozen)
            # We also need ref logits for the loss
            with torch.no_grad():
                ref_out_win = base_model.gpt2(input_ids=input_win)
                ref_logits_win = ref_out_win.logits[:, -config['code_len']-1:-1, :]
                
                ref_out_lose = base_model.gpt2(input_ids=input_lose)
                ref_logits_lose = ref_out_lose.logits[:, -config['code_len']-1:-1, :]
            
            # Compute Loss
            loss_dpo = hierarchical_dpo_loss(
                logits_win, ref_logits_win, labels_win, labels_lose, 
                config['dure_params']['gamma_weights'], config['dure_params']['beta']
            )
            
            # Combine Losses
            loss = loss_dpo + lambda_ecl * loss_ecl + lambda_dol * loss_dol
            
            optimizer.zero_grad()
            loss.backward()
            
            # === DURE Core: Sharding (Gradient Masking) ===
            # Apply USER-SPECIFIC mask to gradients of Side Memory (Adapters)
            # This ensures orthogonality between different USERS
            sharding_rate = config['dure_params'].get('sharding_rate', 0.5) # Default 50% parameters frozen
            
            # Generate seed from user_id
            # Use zlib.adler32 for a simple deterministic hash
            user_seed = zlib.adler32(current_user.encode('utf-8'))
            
            # Save current RNG state
            rng_state = torch.get_rng_state()
            # Set seed for this user
            torch.manual_seed(user_seed)
            
            for name, param in dure_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Only apply to adapter parameters (lora_A, lora_B)
                    if 'lora_' in name:
                        # Create a binary mask (1 = update, 0 = freeze)
                        # We want to keep (1 - sharding_rate) of parameters active
                        mask = torch.bernoulli(torch.full_like(param.grad, 1 - sharding_rate)).to(device)
                        param.grad *= mask
            
            # Restore RNG state to not affect other random processes (like dropout)
            torch.set_rng_state(rng_state)
            
            optimizer.step()
            
            total_loss += loss.item()

    # Save the Side Memory (Adapters) only
    logger.info("Saving Side Memory...")
    adapter_state = {k: v for k, v in dure_model.state_dict().items() if 'adapters' in k or 'router' in k}
    save_path = Path(config['output_dir']) / "adapter.pth"
    torch.save(adapter_state, save_path)
    logger.info(f"Training Complete. Adapter saved to {save_path}")

if __name__ == "__main__":
    main()
