import torch
import torch.nn.functional as F

def compute_log_prob(logits, labels):
    """
    logits: (batch, seq_len, vocab_size)
    labels: (batch, seq_len)
    Returns: (batch, seq_len) log probabilities of the labels
    """
    # logits: [B, L, V]
    # labels: [B, L]
    
    # Shift logits and labels if necessary? 
    # Usually logits at t predict t+1.
    # If labels are the target sequence, and logits are predictions for them.
    # We assume logits[:, i, :] predicts labels[:, i].
    
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather the log prob of the true label
    # labels.unsqueeze(-1): [B, L, 1]
    gathered_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    return gathered_log_probs

def hierarchical_dpo_loss(policy_logits, ref_logits, labels_win, labels_lose, gamma_weights, beta=0.1):
    """
    policy_logits: (batch, seq_len, vocab_size) - from Side Memory model
    ref_logits: (batch, seq_len, vocab_size) - from Frozen Main model
    labels_win: (batch, seq_len) - Semantic IDs of Win item
    labels_lose: (batch, seq_len) - Semantic IDs of Lose item (Dirty item)
    gamma_weights: list or tensor of weights for each level [gamma_1, gamma_2, ...]
    beta: DPO temperature
    """
    # We assume seq_len corresponds to the number of levels in Semantic ID (e.g. 3)
    # If logits include history, we must slice them to only the target part.
    # The caller should ensure logits and labels are aligned to the target sequence.
    
    num_levels = len(gamma_weights)
    
    # Ensure shapes match
    # policy_logits should be [B, num_levels, V]
    # labels should be [B, num_levels]
    
    loss = 0
    
    # Compute log probs for the whole sequence
    # policy_log_probs_win: [B, num_levels]
    policy_log_probs_win = compute_log_prob(policy_logits, labels_win)
    ref_log_probs_win = compute_log_prob(ref_logits, labels_win)
    
    policy_log_probs_lose = compute_log_prob(policy_logits, labels_lose)
    ref_log_probs_lose = compute_log_prob(ref_logits, labels_lose)
    
    # Iterate over levels
    for k in range(num_levels):
        # Accumulate log probs up to level k?
        # The formula says: log pi(z_k | H, z_<k)
        # This is exactly what the model outputs at step k (conditioned on previous tokens).
        # So we just take the k-th term.
        
        log_prob_policy_win_k = policy_log_probs_win[:, k]
        log_prob_ref_win_k = ref_log_probs_win[:, k]
        
        log_prob_policy_lose_k = policy_log_probs_lose[:, k]
        log_prob_ref_lose_k = ref_log_probs_lose[:, k]
        
        # DPO Log Ratio
        # log(pi_side / pi_ref) = log_pi_side - log_pi_ref
        ratio_win = log_prob_policy_win_k - log_prob_ref_win_k
        ratio_lose = log_prob_policy_lose_k - log_prob_ref_lose_k
        
        # Sigmoid Loss
        # L = - log sigmoid ( beta * (ratio_win - ratio_lose) )
        level_loss = -F.logsigmoid(beta * (ratio_win - ratio_lose))
        
        # Weighted sum
        # level_loss is [B], we average over batch later or now?
        # Usually average over batch.
        loss += gamma_weights[k] * level_loss.mean()
        
    return loss
