# /tokenlization_stage/tokenlization_dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self, npy_file_path: str):
        super().__init__()
        self.embeddings = np.load(npy_file_path)
        self.embeddings = torch.from_numpy(self.embeddings).float()
    
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, index):
        return self.embeddings[index]