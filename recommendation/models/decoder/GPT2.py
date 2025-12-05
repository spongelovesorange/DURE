import torch
import torch.nn as nn
from typing import Any, Dict, List
import transformers

from ..abstract_model import AbstractModel
import sys
from pathlib import Path
# 確保 metrics 模組可以被正確導入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from metrics import recall_at_k, ndcg_at_k

# 從 transformers 導入 GPT-2 相關的配置和模型
GPT2Config = transformers.GPT2Config
GPT2LMHeadModel = transformers.GPT2LMHeadModel

class GPT2(AbstractModel):
    """
    一個仿照 TIGER 介面的 Decoder-Only 生成式模型。
    它使用 GPT-2 架構從零開始訓練，專用於序列推薦任務。
    """
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)
        
        model_params = config['model_params']
        token_params = config['token_params']

        # 1. 創建 GPT-2 的設定
        gpt2config = GPT2Config(
            vocab_size=token_params['vocab_size'],
            # 總長度 = 歷史最大長度 + 目標 code 長度
            n_positions=model_params['max_len'] * config['code_len'] + config['code_len'],
            n_embd=model_params['n_embd'],
            n_layer=model_params['n_layer'],
            n_head=model_params['n_head'],
            n_inner=model_params.get('n_inner', model_params.get('d_ff', 2048)), # 兼容 d_ff
            activation_function=model_params.get('activation_function', 'gelu_new'),
            resid_pdrop=model_params.get('resid_pdrop', 0.1),
            embd_pdrop=model_params.get('embd_pdrop', 0.1),
            attn_pdrop=model_params.get('attn_pdrop', 0.1),
            layer_norm_epsilon=float(model_params.get('layer_norm_epsilon', 1e-5)),
            initializer_range=model_params.get('initializer_range', 0.02),
            # 指定特殊 token，生成時會用到
            eos_token_id=token_params['eos_token_id'],
            pad_token_id=token_params['pad_token_id'],
        )

        # 2. 實例化 GPT2LMHeadModel (for Language Modeling)
        self.gpt2 = GPT2LMHeadModel(config=gpt2config)
        self.n_params_str = self._calculate_n_parameters()

    @property
    def task_type(self) -> str:
        return 'generative'

    @property
    def n_parameters(self) -> str:
        return self.n_params_str

    def _calculate_n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.gpt2.get_input_embeddings().parameters())
        return (
            f'# Embedding parameters: {emb_params:,}\n'
            f'# Non-embedding parameters: {total_params - emb_params:,}\n'
            f'# Total trainable parameters: {total_params:,}\n'
        )
    
    def forward(self, batch: Dict) -> Dict:
        """
        【已修正】为 Decoder-Only 模型准备输入。
        核心思想：将 history 和 labels 拼接成一个长序列进行自回归训练。
        """
        history_ids = batch['input_ids']      # (B, L_hist_flat)
        target_ids = batch['labels']          # (B, L_target)
        history_mask = batch['attention_mask'] # (B, L_hist_flat)
        
        # 1. 拼接输入序列: [history_tokens, target_tokens]
        combined_ids = torch.cat([history_ids, target_ids], dim=1)
        
        # 2. 创建拼接后的 attention mask
        target_mask = torch.ones_like(target_ids)
        combined_mask = torch.cat([history_mask, target_mask], dim=1)

        # 3. ✅ 【关键修正】创建用于计算 loss 的 labels
        #    Labels 应该是 Combined_ids 的一个副本
        #    我们只在 "非padding" 的 token 上计算 loss
        
        # 复制 combined_ids 作为基础
        combined_labels = combined_ids.clone() 
        
        # ✅ 使用 combined_mask 来掩盖掉所有的 padding token
        #    (注意：你的 tokenizer 已经确保 history_ids 的 padding token ID 是 0)
        combined_labels[combined_mask == 0] = -100

        # ❌ 【BUG】原来的错误做法 (只在SFT时使用)
        # history_labels = torch.full_like(history_ids, -100)
        # combined_labels = torch.cat([history_labels, target_ids], dim=1)

        # 4. 传给 GPT-2 模型
        outputs = self.gpt2(
            input_ids=combined_ids,
            attention_mask=combined_mask,
            labels=combined_labels  # ✅ 使用修正后的 labels
        )
        return outputs

    def generate(self, **kwargs: Any) -> torch.Tensor:
        """執行 GPT-2 的標準生成。"""
        # generate 只需要 history (input_ids) 作為 prompt
        return self.gpt2.generate(**kwargs)

    def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
        """
        【已修正】此版本返回指标的总和 (sum) 和批次大小 ('count')，
        以配合 trainer.py 中的正确平均值计算。
        """
        beam_size = self.config['evaluation_params']['beam_size']
        code_len = self.config['code_len']

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'] # (B, L_target) - Target Codes
        device = input_ids.device
        
        # ✅ 获取真实的批次大小
        batch_size = labels.shape[0] 

        # 1. 生成 (不变)
        preds = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            max_new_tokens=code_len,
            early_stopping=False,
            pad_token_id=self.config['token_params']['pad_token_id'],
            eos_token_id=None
        )
        
        # 2. 后处理 (不变)
        generated_part = preds[:, input_ids.shape[1]:]
        preds_reshaped = generated_part.view(batch_size, beam_size, -1)
        
        # 3. 计算命中 (不变)
        pos_index = self._calculate_pos_index(preds_reshaped, labels, maxk=beam_size)
        # pos_index 不需要移动到 device，因为它是在 CPU 上计算并用于后续 CPU 计算的
        
        # 4. 计算指标总和
        batch_metrics = {}
        for k in topk_list:
            # ✅ 计算批次内的总和 (sum)，而不是平均值 (mean)
            # 注意：pos_index 已经是 CPU tensor
            recall_sum = recall_at_k(pos_index, k).sum().item() 
            ndcg_sum = ndcg_at_k(pos_index, k).sum().item()
            
            # 存储总和
            batch_metrics[f'Recall@{k}'] = recall_sum 
            batch_metrics[f'NDCG@{k}'] = ndcg_sum
            
        # ✅ 5. 添加批次大小 'count'
        batch_metrics['count'] = float(batch_size) 
          
        # 返回包含 count 和 指标总和 的字典
        return batch_metrics
  
    @staticmethod
    def _calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor, maxk: int) -> torch.Tensor:
        """
        【與 TIGER 共享的評估邏輯】
        假設 code 總是包含 L-1 個語義層和最後 1 個重複層。
        """
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        B, _, L_pred = preds.shape
        L_label = labels.shape[1]

        # 如果生成長度不足（例如提前遇到 EOS），用 padding 補齊
        if L_pred < L_label:
            padding = torch.zeros((B, maxk, L_label - L_pred), dtype=preds.dtype)
            preds = torch.cat([preds, padding], dim=2)
        # 如果生成長度過長，截斷
        elif L_pred > L_label:
            preds = preds[:, :, :L_label]
        
        pos_index = torch.zeros((B, maxk), dtype=torch.bool)
        for i in range(B):
            gt = labels[i]
            gt_semantic = gt[:-1].tolist()
            gt_dup  = int(gt[-1].item())

            for j in range(maxk):
                pj = preds[i, j]
                pj_semantic = pj[:-1].tolist()
                pj_dup  = int(pj[-1].item())

                if pj_semantic == gt_semantic and pj_dup == gt_dup:
                    pos_index[i, j] = True
                    break
        return pos_index

