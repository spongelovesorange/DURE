# models/tiger.py (遵守新契约)

from typing import Any, Dict, List, Optional
import torch
import logging
logger = logging.getLogger(__name__)
import transformers

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from recommendation.metrics import recall_at_k, ndcg_at_k
from recommendation.models.generation.prefix_tree import Trie
from recommendation.models.abstract_model import AbstractModel



T5ForConditionalGeneration = transformers.T5ForConditionalGeneration
T5Config = transformers.T5Config

# TIGER 现在继承自我们定义好的 ABC
class TIGER(AbstractModel):
  
  def __init__(
      self, 
      config: Dict[str, Any], 
      prefix_trie: Optional[Trie] = None
  ):
    super().__init__(config)
    # ... (初始化代码完全不变)
    model_params = config['model_params']
    token_params = config['token_params']
    t5config = T5Config(
        **model_params,
        **token_params,
        decoder_start_token_id=0
    )
    self.t5 = T5ForConditionalGeneration(config=t5config)
    self.t5.resize_token_embeddings(config['token_params']['vocab_size'])
    self.n_params_str = self._calculate_n_parameters() # 在初始化时计算一次

    self.prefix_trie_fn = None
    if prefix_trie is not None:
        self.prefix_trie_fn = prefix_trie.get_allowed_next_tokens
        logger.info("TIGER 模型已成功加载前缀树 (Prefix Trie)。")
    else:
        logger.info("TIGER 模型未加载前缀树 (Prefix Trie)。")

  @property
  def task_type(self) -> str:
        return 'generative'

  @property
  def n_parameters(self) -> str:
    # (可以覆盖基类方法以提供更详细的参数信息)
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
  
  # --- 遵守契约 ---

  def forward(self, batch: Dict) -> Dict:
        """
        【已修正】此版本會先從通用的 batch 中，只挑選出 T5 模型需要的參數，
        然後再進行傳遞，以避免 TypeError。
        """
        # 1. 定義 T5 模型在訓練時認識的參數名稱
        t5_known_args = {
            'input_ids', 
            'attention_mask', 
            'labels'
        }
        
        # 2. 建立一個只包含 T5 認識參數的新字典
        t5_inputs = {key: value for key, value in batch.items() if key in t5_known_args}
        
        # 3. 將這個「乾淨」的字典傳遞給 T5 模型
        return self.t5(**t5_inputs)

  # ✅ 关键修改 3: generate 函数注入 prefix_trie_fn
  def generate(self, **kwargs: Any) -> torch.Tensor:
    """
    【已修改】执行 T5 的标准生成。
    如果Trie存在，则自动将其注入到 'generate' 的参数中。
    """
    
    # 如果Trie已初始化，则将其 "prefix_allowed_tokens_fn" 添加到kwargs
    if self.prefix_trie_fn is not None:
        # 只有在 kwargs 中没有手动提供该函数时才注入
        kwargs.setdefault('prefix_allowed_tokens_fn', self.prefix_trie_fn)
        
    return self.t5.generate(**kwargs)

  def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
    """
    【已修正】封装 TIGER 专属的评估逻辑。
    - 此版本现在返回指标的 *总和 (Sum)* 和 *批次大小 (Count)*
    - 这将配合 trainer.py 中修正后的聚合逻辑。
    """
    # 从 config 中获取评估参数
    beam_size = self.config['evaluation_params']['beam_size']
    code_len = self.config['code_len']

    input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
    device = input_ids.device
    
    # ✅ 1. 获取批次大小
    batch_size = input_ids.shape[0]

    # 1. 生成 (调用自身的 generate)
    preds = self.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        num_beams=beam_size, num_return_sequences=beam_size,
        max_new_tokens=code_len, early_stopping=False
    )
    
    # 2. 后处理
    preds = preds[:, 1:1 + code_len].view(batch_size, beam_size, -1)
    
    # 3. 计算命中 (专属逻辑)
    pos_index = self._calculate_pos_index(preds, labels, maxk=beam_size).to(device)

    # 4. 计算指标 (通用逻辑)
    batch_metrics = {}
    
    # ✅ 2. 将 'count' 添加到返回的字典中
    batch_metrics['count'] = batch_size
    
    for k in topk_list:
        # ✅ 3. 计算 .sum() 而不是 .mean()
        recall_sum = recall_at_k(pos_index, k).sum().item()
        ndcg_sum = ndcg_at_k(pos_index, k).sum().item()
        
        batch_metrics[f'Recall@{k}'] = recall_sum
        batch_metrics[f'NDCG@{k}'] = ndcg_sum
          
    return batch_metrics
  
  # --- TIGER 专属的内部方法 ---
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