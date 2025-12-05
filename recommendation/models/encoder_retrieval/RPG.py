# 檔案路徑: recommendation/models/retrieval/RPG.py (最終還原版 v4)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
import numpy as np
from typing import Dict, List
import logging

from ..abstract_model import AbstractModel 
import sys
from pathlib import Path
# 確保 metrics 和 utils/dataset (如果 item2code 在那裡) 可導入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from metrics import recall_at_k, ndcg_at_k
try:
    from utils import item2code
except ImportError:
    try: from dataset import item2code
    except ImportError: raise ImportError("item2code function not found.")

logger = logging.getLogger(__name__)

class ResBlock(nn.Module):
    def __init__(self, hidden_size): 
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None: torch.nn.init.zeros_(self.linear.bias)
        self.act = nn.SiLU()
    def forward(self, x): return x + self.act(self.linear(x))

class RPG(AbstractModel):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        model_params = self.config['model_params']; token_params = self.config['token_params']
        
        # 1. 使用 item2code 構建 item_id -> offset_token_ids 映射
        self.logger.info("Building item_id -> offset_token_ids map...")
        try:
            item_to_code_map, _ = item2code(
                config['code_path'], config['vocab_sizes'], config['bases']
            )
        except Exception as e: self.logger.error(f"Failed to build item_to_code map: {e}"); raise
        
        # 將 map 轉為 Tensor (N+1, L_code)，放 CPU
        max_item_id = max(item_to_code_map.keys()) if item_to_code_map else 0
        self.n_items = max_item_id + 1 
        self.code_len = config['code_len']
        pad_token_id = config['token_params']['pad_token_id']
        self.item_id2tokens_cpu = torch.full((self.n_items, self.code_len), pad_token_id, dtype=torch.long)
        for item_id, tokens in item_to_code_map.items():
            if 0 < item_id < self.n_items: # 確保 item_id 在範圍內且 > 0
                 if len(tokens) == self.code_len: self.item_id2tokens_cpu[item_id] = torch.LongTensor(tokens)
                 else: self.logger.warning(f"Item {item_id} code length mismatch. Skipping.")
            # else: self.logger.warning(f"Item ID {item_id} out of range. Skipping.")
        self.item_id2tokens = None # Lazy move to device
        self.logger.info(f"Built item_id2tokens tensor, shape: {self.item_id2tokens_cpu.shape}")
        # (驗證 codebook 值的程式碼可以保留或移除)

        # 2. 初始化 GPT-2 Config & Model
        gpt2config = GPT2Config(
            vocab_size=token_params['vocab_size'], n_positions=model_params['max_len'],
            n_embd=model_params['n_embd'], n_layer=model_params['n_layer'], n_head=model_params['n_head'],
            n_inner=model_params['n_inner'], activation_function=model_params['activation_function'],
            resid_pdrop=model_params['resid_pdrop'], embd_pdrop=model_params['embd_pdrop'],
            attn_pdrop=model_params['attn_pdrop'], layer_norm_epsilon=float(model_params['layer_norm_epsilon']),
            initializer_range=model_params['initializer_range'], eos_token_id=token_params['eos_token_id'],
            pad_token_id=token_params['pad_token_id']
        )
        self.gpt2 = GPT2Model(gpt2config) # No pre-trained weights
        self.gpt2.resize_token_embeddings(token_params['vocab_size']) 
        self.logger.info("GPT-2 model initialized and embeddings resized.")

        # 3. 初始化 Prediction Heads
        self.n_pred_head = self.config['code_len']
        # ✅ 使用 ModuleList, 但 forward 中應用方式改變
        self.pred_heads = nn.ModuleList([ResBlock(model_params['n_embd']) for _ in range(self.n_pred_head)]) 

        # 4. 初始化其他
        self.temperature = model_params['temperature']
        # ✅ 使用原始的 ignore_index
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100) 
        self._debug_printed_forward = False # Debug flag

    @property
    def task_type(self) -> str: return 'retrieval'

    @property
    def n_parameters(self) -> str:
        """ (Helper property to get parameter count) """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return (
            f'# Embedding parameters: {emb_params:,}\n'
            f'# Non-embedding parameters: {total_params - emb_params:,}\n'
            f'# Total trainable parameters: {total_params:,}\n'
        )

    def _ensure_item_id2tokens_on_device(self, device):
         if self.item_id2tokens is None or self.item_id2tokens.device != device:
             self.item_id2tokens = self.item_id2tokens_cpu.to(device)

    # ==================== FORWARD (核心修正) ====================
    def forward(self, batch: Dict) -> Dict:
        """
        【最終還原版 v4 - 嚴格對齊原始訓練目標】
        在 *整個序列* 上應用 Head 並使用 label_mask 選擇狀態和目標進行 Loss 計算。
        """
        input_item_ids = batch['input_ids']     # (B, L_item) 1-based IDs + padding 0
        attention_mask = batch['attention_mask'] # (B, L_item) Item-level Mask
        labels_seq = batch['labels_seq']        # (B, L_item) Target ID sequence (-100 padding, 0-based ID)
        batch_size = input_item_ids.shape[0]
        current_device = input_item_ids.device

        self._ensure_item_id2tokens_on_device(current_device)

        # --- 1. Item ID -> Item Embedding ---
        try: input_tokens = self.item_id2tokens[input_item_ids] # (B, L_item, L_code)
        except IndexError as e: self.logger.error(f"IndexError item_id2tokens lookup! Error: {e}"); raise
        token_embs = self.gpt2.wte(input_tokens)           # (B, L_item, L_code, D)
        input_embs = token_embs.mean(dim=2)              # (B, L_item, D)

        # --- 2. GPT-2 ---
        outputs = self.gpt2(inputs_embeds=input_embs, attention_mask=attention_mask)
        gpt2_output_states = outputs.last_hidden_state # (B, L_item, D)

        # --- 3. ✅ 應用 Prediction Heads 到整個序列 ---
        # 這裡我們需要模仿原始碼的結果形狀 (B, L_item, L_code, D)
        # 原始碼: final_states = [self.pred_heads[i](outputs.last_hidden_state)... torch.cat(..., dim=-2)]
        # 這表示每個 head 獨立作用在 gpt2_output_states 上
        head_outputs = [head(gpt2_output_states) for head in self.pred_heads] # List of (B, L_item, D)
        final_states = torch.stack(head_outputs, dim=2) # Shape: (B, L_item, L_code, D)

        # --- 4. ✅ 計算 Loss (嚴格模仿原始 RPG) ---
        # 創建 mask，標記哪些位置需要計算 Loss (label 不是 -100)
        label_mask = labels_seq.view(-1) != -100 # Shape: (B * L_item,)
        valid_indices_flat = label_mask.nonzero(as_tuple=True)[0]
        
        # 如果沒有有效標籤
        if valid_indices_flat.numel() == 0: 
            return {'loss': torch.tensor(0.0, device=current_device, requires_grad=True)}
            
        # 獲取 *對應時間步* 的模型輸出表示
        # .view(-1,...) 將 (B, L_item, L_code, D) -> (B*L_item, L_code, D)
        # 然後用 label_mask 篩選
        selected_states = final_states.view(-1, self.n_pred_head, self.config['model_params']['n_embd'])[label_mask]
        selected_states = F.normalize(selected_states, dim=-1) # (Num_Valid, L_code, D)

        # 獲取目標 Item IDs (0-based)
        valid_target_item_ids_0based = labels_seq.view(-1)[label_mask] # (Num_Valid,)
        
        # 查找目標 Item 的 Code Tokens (Global Offset)
        # 增加邊界檢查
        max_target_id = valid_target_item_ids_0based.max().item(); min_target_id = valid_target_item_ids_0based.min().item()
        if max_target_id + 1 >= self.item_id2tokens.shape[0] or min_target_id < 0:
             self.logger.error(f"FATAL: Target Item ID out of bounds! Range: [{min_target_id}, {max_target_id}], Codebook items: {self.item_id2tokens.shape[0]-1}. Check DataLoader."); return {'loss': torch.tensor(1000.0, device=current_device, requires_grad=True)}
        # 使用 0-based ID + 1 作為 1-based index 查找
        token_labels = self.item_id2tokens[valid_target_item_ids_0based + 1] # (Num_Valid, L_code)

        # 準備 global token embeddings (用於計算相似度)
        token_emb = self.gpt2.wte.weight; token_emb_norm = F.normalize(token_emb, dim=-1)
        bases = torch.tensor(self.config['bases'], device=current_device); vocab_sizes = torch.tensor(self.config['vocab_sizes'], device=current_device)
        
        # 判斷是否可以使用 chunk (保持不變)
        all_same_size = all(vs == vocab_sizes[0] for vs in vocab_sizes); use_chunk = False
        if all_same_size and len(vocab_sizes) == self.n_pred_head:
            codebook_size_k = vocab_sizes[0].item(); start_idx = 1
            max_valid_code_token_id = bases[-1] + vocab_sizes[-1]; end_idx = min(start_idx + self.n_pred_head * codebook_size_k, token_emb.shape[0])
            usable_token_embs = token_emb_norm[start_idx : end_idx]
            if usable_token_embs.shape[0] == self.n_pred_head * codebook_size_k:
                token_embs_chunks = torch.chunk(usable_token_embs, self.n_pred_head, dim=0); use_chunk = True
            # else: self.logger.warning("Chunk fallback.")
        
        losses = []
        # --- (移除 Debug 打印) ---

        for i in range(self.n_pred_head):
            # 獲取第 i 層的 Codebook Embeddings
            if use_chunk: token_embs_i = token_embs_chunks[i] # (K, D)
            else: start_idx = bases[i] + 1; end_idx = min(start_idx + vocab_sizes[i], token_emb.shape[0]); token_embs_i = token_emb_norm[start_idx:end_idx] # (K, D)
            
            # 獲取第 i 層的 User Representation (來自對應時間步)
            user_repr_i = selected_states[:, i, :] # (Num_Valid, D)
            
            # 計算 Logits
            logits_i = torch.matmul(user_repr_i, token_embs_i.T) / self.temperature # (Num_Valid, K)
            
            # 獲取第 i 層的 Target Labels (0-based)
            labels_i = token_labels[:, i] - bases[i] - 1 # (Num_Valid,)

            # 最終邊界檢查 (保持)
            if not (torch.all(labels_i >= 0) and torch.all(labels_i < logits_i.shape[1])):
                 invalid_mask = (labels_i < 0) | (labels_i >= logits_i.shape[1]); invalid_labels = labels_i[invalid_mask]; corresponding_global_tokens = token_labels[:, i][invalid_mask]
                 self.logger.error(f"FATAL: Head {i} labels out of bounds! Logits Dim 1: {logits_i.shape[1]}. Invalid labels: {invalid_labels.cpu().numpy()}. Global tokens: {corresponding_global_tokens.cpu().numpy()}. Skipping loss.")
                 losses.append(torch.tensor(100.0, device=current_device, requires_grad=True)); continue
            
            # ✅ 使用原始的 self.loss_fct (帶 ignore_index)
            # 雖然 label_mask 已經過濾了 -100，但 CrossEntropyLoss 本身處理 0-based 索引
            # 這裡 logits_i 是 (Num_Valid, K)，labels_i 是 (Num_Valid,)，符合要求
            losses.append(self.loss_fct(logits_i, labels_i))

        if losses: 
            loss = torch.mean(torch.stack(losses))
            if not torch.isfinite(loss):
                self.logger.error("FATAL: Mean loss is NaN or Inf!")
                loss = torch.tensor(1000.0, device=current_device, requires_grad=True) 
        else: 
            # This case should ideally not happen if label_mask works correctly
            # and at least one label is not -100 in the batch.
            self.logger.warning("No losses were calculated for this batch (maybe all labels were -100?). Returning 0 loss.")
            loss = torch.tensor(0.0, device=current_device, requires_grad=True) 

        return {'loss': loss}

    # --- _generate_ranklist 和 evaluate_step 保持不變 (它們只用最後狀態) ---
    def _generate_ranklist(self, batch: Dict, topk: int) -> torch.Tensor:
        """
        (v4 - 保持不變) 
        為評估生成 topk 排序列表。
        只使用 *最後* 的隱藏狀態來表示用戶。
        """
        input_item_ids = batch['input_ids']; attention_mask = batch['attention_mask']; batch_size = input_item_ids.shape[0]
        current_device = input_item_ids.device
        self._ensure_item_id2tokens_on_device(current_device)
        
        # --- 1. Item ID -> Item Embedding ---
        input_tokens = self.item_id2tokens[input_item_ids]; token_embs = self.gpt2.wte(input_tokens); input_embs = token_embs.mean(dim=2)
        
        # --- 2. GPT-2 ---
        outputs = self.gpt2(inputs_embeds=input_embs, attention_mask=attention_mask)
        
        # --- 3. 获取 *最后* 的隐藏状态（保持不变） ---
        seq_lens = attention_mask.sum(dim=1)
        valid_lens = torch.clamp(seq_lens - 1, min=0)
        max_seq_len_in_batch = outputs.last_hidden_state.shape[1]
        valid_lens = torch.min(valid_lens, torch.tensor(max_seq_len_in_batch - 1, device=current_device))
        last_hidden_states = outputs.last_hidden_state[torch.arange(batch_size, device=current_device), valid_lens]  # (B, D)

        # --- 4. 应用 Heads 到 *最后* 状态（与训练一致） ---
        final_states = [head(last_hidden_states) for head in self.pred_heads]  # list of (B, D)
        final_states = torch.stack(final_states, dim=1)  # (B, L_code, D)
        final_states = F.normalize(final_states, dim=-1)

        # --- 5. 预取 token embedding，并按 codebook 分段 ---
        token_emb = self.gpt2.wte.weight  # (V, D)
        token_emb = F.normalize(token_emb, dim=-1)

        bases = torch.tensor(self.config['bases'], device=current_device)
        vocab_sizes = torch.tensor(self.config['vocab_sizes'], device=current_device)
        assert self.n_pred_head == len(vocab_sizes), "n_pred_head 与 vocab_sizes 长度不一致"

        all_same = bool(torch.all(vocab_sizes == vocab_sizes[0]).item())
        token_embs_per_digit = []
        if all_same and len(vocab_sizes) == self.n_pred_head:
            K = int(vocab_sizes[0].item())
            start = 1  # 跳过 padding 0
            end = start + self.n_pred_head * K
            assert end <= token_emb.shape[0], "token_emb 词表不足以切成等长分块"
            chunked = torch.chunk(token_emb[start:end], self.n_pred_head, dim=0)
            token_embs_per_digit = list(chunked)  # 每个 (K, D)
        else:
            for i in range(self.n_pred_head):
                s = int(bases[i].item()) + 1
                e = s + int(vocab_sizes[i].item())
                assert e <= token_emb.shape[0], f"digit {i} 的 vocab 切片越界: [{s},{e})"
                token_embs_per_digit.append(token_emb[s:e])  # (K_i, D)

        # --- 6. 计算每位 digit 的 log_softmax 概率（与训练一致） ---
        log_probs_per_digit = []
        for i in range(self.n_pred_head):
            logits_i = torch.matmul(final_states[:, i, :], token_embs_per_digit[i].T) / self.temperature  # (B, K_i)
            log_probs_per_digit.append(F.log_softmax(logits_i, dim=-1))  # (B, K_i)

        # --- 7. 按 item 的每位 token 索引 gather 出 log_prob 并求和，得到 item 分数 ---
        all_item_codes = self.item_id2tokens[1:]  # (N_items, L_code) 1-based 全局 token id
        per_digit_indices = []
        for i in range(self.n_pred_head):
            if all_same:
                K = int(vocab_sizes[0].item())
                base_i = i * K
                idx_i = all_item_codes[:, i] - base_i - 1  # (N_items,)
            else:
                base_i = int(bases[i].item())
                idx_i = all_item_codes[:, i] - base_i - 1
            assert torch.all((idx_i >= 0) & (idx_i < log_probs_per_digit[i].shape[1])), f"digit {i} 索引越界"
            per_digit_indices.append(idx_i)

        item_scores = None
        for i in range(self.n_pred_head):
            lp = log_probs_per_digit[i]  # (B, K_i)
            idx = per_digit_indices[i].unsqueeze(0).expand(batch_size, -1)  # (B, N_items)
            gathered = torch.gather(lp, dim=1, index=idx)  # (B, N_items)
            item_scores = gathered if item_scores is None else (item_scores + gathered)

        # --- 8. 取 TopK（注意 item_id 要 +1 还原 1-based） ---
        _, topk_indices = torch.topk(item_scores, k=topk, dim=1)  # (B, k)
        return topk_indices + 1


    # ==================== EVALUATE (核心修正) ====================
    def evaluate_step(self, batch: Dict, topk_list: List[int]) -> Dict[str, float]:
        """
        【最終還原版 v4 - 已修正】
        此版本返回指標的 *总和 (sum)* 和批次大小 ('count')，
        以配合 trainer.py 中的正确平均值计算。
        """
        max_k = max(topk_list)
        
        # 1. 獲取 Ranklist (1-based IDs) 和 Targets (0-based IDs)
        # _generate_ranklist 返回 1-based item IDs
        ranked_item_indices = self._generate_ranklist(batch, topk=max_k) # (B, max_k)
        
        # 從 dataloader 獲取 0-based target IDs
        target_ids = batch['target_ids'].unsqueeze(1) # (B, 1), 0-based IDs
        
        # ✅ 獲取真實的批次大小
        batch_size = target_ids.shape[0]

        # 2. 計算命中 (boolean tensor)
        # 將 1-based ranked list 轉換為 0-based, 然後與 0-based target 比較
        hits = ((ranked_item_indices - 1) == target_ids).cpu() # (B, max_k)

        # 3. 計算指標總和
        batch_metrics = {}
        for k in topk_list:
            pos_index_k = hits[:, :k] # (B, k)
            
            if pos_index_k.numel() > 0:
                # ✅ 計算批次內的總和 (sum)，而不是平均值 (mean)
                recall_sum = recall_at_k(pos_index_k, k).sum().item() 
                ndcg_sum = ndcg_at_k(pos_index_k, k).sum().item()
            else:
                recall_sum = 0.0
                ndcg_sum = 0.0
                
            # 存儲總和
            batch_metrics[f'Recall@{k}'] = recall_sum 
            batch_metrics[f'NDCG@{k}'] = ndcg_sum
            
        # ✅ 4. 添加批次大小 'count'
        batch_metrics['count'] = float(batch_size) 
          
        # 返回包含 count 和 指標總和 的字典
        return batch_metrics