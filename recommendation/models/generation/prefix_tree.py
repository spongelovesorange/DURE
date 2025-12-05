# models/generation/prefix_trie.py

import torch
import logging
from typing import List, Dict, Any, Iterable

logger = logging.getLogger(__name__)

class Trie:
    """
    一个用于Hugging Face 'generate'函数的前缀树 (Trie)。
    它实现了 'prefix_allowed_tokens_fn' 所需的接口。
    """
    
    # 定义一个特殊的Token来标记序列的结束
    END_TOKEN = -1

    def __init__(self, eos_token_id: int):
        """
        初始化Trie。
        
        Args:
            eos_token_id (int): 当一个序列在Trie中完成时，
                                我们将强制模型生成这个Token。
        """
        self.root: Dict[int, Any] = {}
        self.eos_token_id = eos_token_id
        logger.info(f"Trie a初始化，将使用 EOS Token ID: {eos_token_id}")

    def add_sequence(self, sequence: List[int]):
        """
        将一个合法的Token序列添加到Trie中。
        
        例如: [100, 205, 301, 5] (c0, c1, c2, dup)
        """
        node = self.root
        for token in sequence:
            if token not in node:
                node[token] = {}
            node = node[token]
        
        # 在序列的末尾标记 "END"
        node[self.END_TOKEN] = True

    def get_allowed_next_tokens(self, batch_id: int, input_ids: torch.Tensor) -> List[int]:
        """
        Hugging Face 'generate' 会调用的核心函数。
        
        Args:
            batch_id (int): 当前批次中的索引 (我们通常忽略)。
            input_ids (torch.Tensor): *已经* 生成的Token序列。
                                      对于T5/TIGER，它通常以 [0] (pad/start) 开头。
        
        Returns:
            List[int]: 接下来 *允许* 生成的Token ID列表。
        """
        
        # 1. 获取当前序列并清理 (去掉T5的起始Token 0)
        current_sequence = input_ids.tolist()
        if current_sequence and current_sequence[0] == 0:
            current_sequence = current_sequence[1:]

        # 2. 在Trie中遍历当前序列
        node = self.root
        for token in current_sequence:
            if token in node:
                node = node[token]
            else:
                # 如果走到了一个无效路径 (理论上不应发生，除非beam search出错了)
                # 安全起见，返回EOS
                logger.warning(f"Trie: 无效的中间路径 {current_sequence}")
                return [self.eos_token_id]

        # 3. 决定下一步能走哪些Token
        if self.END_TOKEN in node:
            # 这条路径已经是一个完整的、合法的Item Code了。
            # 强制模型停止生成 (只允许生成EOS)。
            return [self.eos_token_id]
        else:
            # 序列尚未完成，返回所有可能的下一步Token。
            # (node.keys() 包含了所有合法的 c1, c2, dup... 等)
            return list(node.keys())


def build_trie_from_codebook(
    token_sequences: Iterable[List[int]], 
    eos_token_id: int
) -> Trie:
    """
    辅助函数：从codebook的Token序列构建Trie。
    
    Args:
        token_sequences (Iterable[List[int]]): 
            所有合法的、完整的Token序列的列表。
            例如: [[100, 205, 301, 5], [100, 206, 302, 6], ...]
        eos_token_id (int): 模型的EOS Token ID。
            
    Returns:
        Trie: 构建完成的前缀树。
    """
    trie = Trie(eos_token_id=eos_token_id)
    count = 0
    for seq in token_sequences:
        trie.add_sequence(seq)
        count += 1
    
    logger.info(f"成功构建前缀树 (Prefix Trie)，共加载 {count} 条合法序列。")
    return trie