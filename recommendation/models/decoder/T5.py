import torch
from typing import Any, Dict, List
import transformers

from ..abstract_model import AbstractModel
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from metrics import recall_at_k, ndcg_at_k

T5ForConditionalGeneration = transformers.T5ForConditionalGeneration
T5Config = transformers.T5Config


class T5(AbstractModel):
    """
    Encoder-Decoder (T5) variant for code-token based generative recommendation.
    - Loads a local T5 checkpoint, then resizes embeddings/LM head to code-token vocab.
    - Uses encoder input = flattened history code tokens, decoder labels = target code tokens.
    - Reuses evaluation logic (beam search over code_len tokens).
    """
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)

        model_params = config['model_params']
        token_params = config['token_params']

        model_name_or_path = model_params.get('model_name_or_path')
        if not model_name_or_path:
            raise ValueError("'model_params.model_name_or_path' is required for T5")

        # Load local T5 and then adapt vocab to our code-token space
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

        # 降低显存：训练阶段禁用 KV Cache；是否启用梯度检查点由配置控制（默认启用）
        try:
            self.t5.config.use_cache = False
            gc_enable = bool(model_params.get('grad_checkpoint', True))
            if gc_enable and hasattr(self.t5, 'gradient_checkpointing_enable'):
                self.t5.gradient_checkpointing_enable()
        except Exception:
            pass

        code_vocab_size = token_params['vocab_size']
        # Resize embeddings and lm head
        self.t5.resize_token_embeddings(code_vocab_size)

        # Ensure special ids align with our code tokenization
        self.t5.config.vocab_size = code_vocab_size
        self.t5.config.pad_token_id = token_params['pad_token_id']
        self.t5.config.eos_token_id = token_params['eos_token_id']
        # Common choice for T5: decoder starts from pad token
        self.t5.config.decoder_start_token_id = token_params['pad_token_id']

        self.code_len = config['code_len']
        self._pad_id = token_params['pad_token_id']
        self._eos_id = token_params['eos_token_id']

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
        emb_params = num_params(self.t5.get_input_embeddings().parameters())
        return (
            f'# Embedding parameters: {emb_params:,}\n'
            f'# Non-embedding parameters: {total_params - emb_params:,}\n'
            f'# Total trainable parameters: {total_params:,}\n'
        )

    def forward(self, batch: Dict) -> Dict:
        input_ids = batch['input_ids']          # (B, L_hist_flat)
        attention_mask = batch['attention_mask']# (B, L_hist_flat)
        labels = batch['labels']                # (B, code_len)

        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def generate(self, **kwargs: Any) -> torch.Tensor:
        kwargs.setdefault('pad_token_id', self._pad_id)
        kwargs.setdefault('eos_token_id', self._eos_id)
        return self.t5.generate(**kwargs)

    def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
        beam_size = self.config['evaluation_params']['beam_size']

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        device = input_ids.device

        batch_size = labels.shape[0]

        preds = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            max_new_tokens=self.code_len,
            early_stopping=False,
        )

        generated_part = preds[:, 0: self.code_len] if preds.shape[1] <= self.code_len else preds[:, -self.code_len:]
        # For enc-dec, generate returns only decoded tokens; reshape into (B, beams, L)
        preds_reshaped = generated_part.view(batch_size, beam_size, -1)

        pos_index = self._calculate_pos_index(preds_reshaped, labels, maxk=beam_size)
        # Compute sums for proper averaging in trainer
        batch_metrics = {'count': float(batch_size)}
        for k in topk_list:
            batch_metrics[f'Recall@{k}'] = recall_at_k(pos_index, k).sum().item()
            batch_metrics[f'NDCG@{k}'] = ndcg_at_k(pos_index, k).sum().item()
        return batch_metrics

    @staticmethod
    def _calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor, maxk: int) -> torch.Tensor:
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        B, K, L_pred = preds.shape
        L_label = labels.shape[1]
        if L_pred < L_label:
            padding = torch.zeros((B, K, L_label - L_pred), dtype=preds.dtype)
            preds = torch.cat([preds, padding], dim=2)
        elif L_pred > L_label:
            preds = preds[:, :, :L_label]

        pos_index = torch.zeros((B, maxk), dtype=torch.bool)
        for i in range(B):
            gt = labels[i]
            gt_semantic = gt[:-1].tolist()
            gt_dup = int(gt[-1].item()) if gt.shape[0] > 0 else 0
            for j in range(maxk):
                pj = preds[i, j]
                pj_semantic = pj[:-1].tolist()
                pj_dup = int(pj[-1].item()) if pj.shape[0] > 0 else -1
                if pj_semantic == gt_semantic and pj_dup == gt_dup:
                    pos_index[i, j] = True
                    break
        return pos_index
