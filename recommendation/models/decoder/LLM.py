# 檔案路徑: recommendation/models/generative/LLM_REC.py (直接數字 ID 版)

import torch
import torch.nn as nn
from typing import Any, Dict, List
import transformers
import logging
from transformers import AutoModelForCausalLM, AutoConfig # 只需要 Config 和 Model

from ..abstract_model import AbstractModel
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from metrics import recall_at_k, ndcg_at_k

logger = logging.getLogger(__name__)

class LLM(AbstractModel):
    """
    【直接數字 ID 版】
    使用預訓練 Decoder-Only LLM 架構 (如 Llama, Qwen)，但拋棄其原始 Embedding，
    直接使用 Code Token Offset ID 作為輸入，訓練一個新的 Embedding 層。
    """
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)
        
        model_params = config['model_params']
        token_params = config['token_params']
        
        # 1. 獲取預訓練模型名稱或路徑
        model_name_or_path = model_params.get('model_name_or_path')
        if not model_name_or_path:
            raise ValueError("設定檔 'model_params' 中必須提供 'model_name_or_path'")
        logger.info(f"載入預訓練模型架構: {model_name_or_path}")

        # 2. 載入預訓練模型的 Config (不需要 Tokenizer)
        llm_config = AutoConfig.from_pretrained(model_name_or_path)
        
        # (可選) 根據您的設定檔覆蓋 LLM Config 的某些參數
        # 例如，如果您想調整層數或隱藏維度 (但不推薦，失去了預訓練的意義)
        # llm_config.n_layer = model_params.get('n_layer', llm_config.n_layer)
        # llm_config.n_embd = model_params.get('n_embd', llm_config.hidden_size) # 注意名稱可能不同

        # 3. 載入預訓練模型 (不包含 LM Head，因為我們要重置 Embedding)
        # 我們只使用其 Transformer 骨架
        # 注意：不同的模型架構可能需要不同的 from_pretrained 方式，
        # AutoModelForCausalLM 通常適用
        # 為了安全起見，我們先載入完整模型
        self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=llm_config)

        # 4. ✅ 關鍵：徹底重置 Embedding 和 LM Head
        # 獲取 Code Token 的總詞彙表大小
        code_vocab_size = token_params['vocab_size']
        logger.info(f"將重置模型 Embedding 層和 LM Head 大小至: {code_vocab_size}")
        self.llm.resize_token_embeddings(code_vocab_size) 
        # 注意：resize_token_embeddings 會自動處理 Embedding 層和輸出層 (LM Head)
        # 新的 Embedding 會被隨機初始化

        # 5. 存儲 PAD 和 EOS ID (來自我們的 Code Token 定義)
        self._pad_id = token_params['pad_token_id']
        self._eos_id = token_params['eos_token_id']

        self.n_params_str = self._calculate_n_parameters()
        self.code_len = config['code_len']

    @property
    def task_type(self) -> str:
        return 'generative'

    @property
    def n_parameters(self) -> str:
        # 現在的 Embedding 層會小得多
        return self._calculate_n_parameters()

    def _calculate_n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        # 注意：現在 get_input_embeddings() 返回的是我們 resize 後的新 embedding 層
        emb_params = num_params(self.llm.get_input_embeddings().parameters())
        return (f'# Embedding parameters: {emb_params:,}\n' f'# Non-embedding parameters: {total_params - emb_params:,}\n' f'# Total trainable parameters: {total_params:,}\n')

    def forward(self, batch: Dict) -> Dict:
        """
        處理方式與 SIMPLE_GPT 完全相同。
        """
        history_ids = batch['input_ids']      # (B, L_hist_flat) - Offset IDs
        target_ids = batch['labels']        # (B, L_target) - Offset IDs
        history_mask = batch['attention_mask'] # (B, L_hist_flat) - Token-level mask
        
        # 1. 拼接輸入序列
        combined_ids = torch.cat([history_ids, target_ids], dim=1)
        
        # 2. 創建拼接後的 attention mask
        target_mask = torch.ones_like(target_ids)
        combined_mask = torch.cat([history_mask, target_mask], dim=1)

        # 3. 創建用於計算 loss 的 labels
        history_labels = torch.full_like(history_ids, -100)
        combined_labels = torch.cat([history_labels, target_ids], dim=1)

        # 4. 傳給 LLM 模型 (現在它接收的是 Offset IDs)
        outputs = self.llm(
            input_ids=combined_ids,
            attention_mask=combined_mask,
            labels=combined_labels
        )
        return outputs

    def generate(self, **kwargs: Any) -> torch.Tensor:
        """執行 LLM 的標準生成 (使用 Code Token IDs)"""
        # 使用我們 Code Token 的 PAD/EOS ID
        kwargs.setdefault("pad_token_id", self._pad_id)
        kwargs.setdefault("eos_token_id", self._eos_id)
        # input_ids 應該是 history_ids (Offset IDs)
        return self.llm.generate(**kwargs)

    def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
        """
        評估邏輯與 SIMPLE_GPT 完全相同。
        """
        beam_size = self.config['evaluation_params']['beam_size']
        code_len = self.code_len

        input_ids = batch['input_ids']         # History Offset IDs
        attention_mask = batch['attention_mask'] # History Token-level Mask
        labels = batch['labels']               # Target Offset IDs
        device = input_ids.device

        # 1. 生成 (輸入是 History Offset IDs)
        preds = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            max_new_tokens=code_len,
            early_stopping=False,
        )
        
        # 2. 後處理 (切掉 prompt 部分)
        generated_part = preds[:, input_ids.shape[1]:] # 得到生成的 Offset IDs
        preds_reshaped = generated_part.view(input_ids.shape[0], beam_size, -1)
        
        # 3. 計算命中 (直接使用 Offset IDs)
        pos_index = self._calculate_pos_index(preds_reshaped, labels, maxk=beam_size)
        pos_index = pos_index.to(device)
        
        # 4. 計算指標
        batch_metrics = {}
        for k in topk_list:
            recall = recall_at_k(pos_index, k).mean().item()
            ndcg = ndcg_at_k(pos_index, k).mean().item()
            batch_metrics[f'Recall@{k}'] = recall
            batch_metrics[f'NDCG@{k}'] = ndcg
            
        return batch_metrics
  
    # _calculate_pos_index 可以保持與 TIGER 一致 (假設有 dup 層)
    @staticmethod
    def _calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor, maxk: int) -> torch.Tensor:
        # ... (與 TIGER 版本相同) ...
        preds = preds.detach().cpu(); labels = labels.detach().cpu()
        B, K, L_pred = preds.shape; L_label = labels.shape[1]
        if L_pred < L_label: padding = torch.full((B, K, L_label - L_pred), 0, dtype=preds.dtype); preds = torch.cat([preds, padding], dim=2)
        elif L_pred > L_label: preds = preds[:, :, :L_label]
        L = L_label
        pos_index = torch.zeros((B, maxk), dtype=torch.bool)
        has_dup_layer = True 
        for i in range(B):
            gt = labels[i]
            if L == 0: continue
            if has_dup_layer and L > 1: gt_semantic, gt_dup = gt[:-1].tolist(), int(gt[-1].item())
            else: gt_semantic, gt_dup = gt.tolist(), 0
            for j in range(min(K, maxk)):
                pj = preds[i, j]
                if has_dup_layer and L > 1: pj_semantic, pj_dup = pj[:-1].tolist(), int(pj[-1].item())
                else: pj_semantic, pj_dup = pj.tolist(), float('inf')
                if pj_semantic == gt_semantic and pj_dup >= gt_dup: pos_index[i, j] = True; break
        return pos_index