import torch
from torch.utils.data import Dataset
import numpy as np
import random
from dataset import pad_or_truncate

class DUREDataset(Dataset):
    def __init__(self, data, item_to_code, max_len, vocab_sizes, bases, pad_token=0):
        """
        data: list of dict {'history': [items], 'target': item}
        item_to_code: dict {item_id: [tokens]}
        """
        self.data = data
        self.item_to_code = item_to_code
        self.max_len = max_len
        self.vocab_sizes = vocab_sizes
        self.bases = bases
        self.pad_token = pad_token
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        history = row['history'] # List of item IDs
        dirty_item = row['target'] # The item to forget
        
        # H_clean is the history without the dirty item (which is the target here)
        # So H_clean is just 'history'.
        
        # We need to tokenize history into Semantic IDs
        # Flatten the history tokens
        history_tokens = []
        for item_id in history:
            if item_id in self.item_to_code:
                history_tokens.extend(self.item_to_code[item_id])
            else:
                # Handle unknown item? Skip or pad?
                pass
                
        # Pad or truncate history
        # Note: GPT2 expects [PAD, PAD, ..., t1, t2]
        # But here we are flattening.
        # Let's assume max_len is in *tokens* or *items*?
        # Usually max_len in config is number of items.
        # But the model input size is max_len * code_len.
        
        code_len = len(self.vocab_sizes)
        max_token_len = self.max_len * code_len
        
        input_ids = pad_or_truncate(history_tokens, max_token_len, self.pad_token)
        
        # Labels Lose: The dirty item tokens
        if dirty_item in self.item_to_code:
            labels_lose = self.item_to_code[dirty_item]
        else:
            # Should not happen if data is consistent
            labels_lose = [0] * code_len
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels_lose': torch.tensor(labels_lose, dtype=torch.long),
            'dirty_item_id': dirty_item # For router or logging
        }

def collate_fn_dure(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels_lose = torch.stack([item['labels_lose'] for item in batch])
    dirty_item_ids = [item['dirty_item_id'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'labels_lose': labels_lose,
        'dirty_item_ids': dirty_item_ids
    }
