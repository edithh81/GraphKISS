import os
import random

import numpy as np
import torch

# def seed_everything(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # if using multi-GPU
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Seed every source of randomness used across data loading and training:
    Python hash, `random`, NumPy, PyTorch (CPU), and all CUDA devices.

    `deterministic=True` also forces cuDNN into deterministic mode and
    enables PyTorch's deterministic-algorithms check. This can be slower
    and makes some ops raise, so keep it off for throughput runs and only
    turn it on when you need exact bit-for-bit reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"[seed_everything] use_deterministic_algorithms failed ({e})")
    else:
        torch.backends.cudnn.benchmark = True


def cal_bpr_loss(n_users, pos, neg_samples, scores):
    """
    Mean BPR loss over all (positive, negative) pairs in the batch.
    """
    losses = []

    for i in range(scores.shape[0]):
        pos_i = torch.as_tensor(pos[i], device=scores.device, dtype=torch.long) - n_users
        neg_i = torch.as_tensor(neg_samples[i], device=scores.device, dtype=torch.long) - n_users

        pos_scores = scores[i][pos_i]
        neg_scores = scores[i][neg_i]
        score_diff = pos_scores.unsqueeze(-1) - neg_scores
        u_loss = -torch.nn.functional.logsigmoid(score_diff).mean()
        losses.append(u_loss)

    return torch.stack(losses).mean()


def _build_hit_matrix(topk_idx: torch.Tensor, pos_padded: torch.Tensor, pos_counts: torch.Tensor) -> torch.Tensor:
    """
    Shared hit matrix for recall / ndcg.

    topk_idx:   [B, K] item-local ids from torch.topk on masked scores.
    pos_padded: [B, P] item-local positives, padded with any invalid value.
    pos_counts: [B]    number of valid positives per user.
    Returns:    [B, K] float hit indicator (1.0 if rank k hits a valid positive).
    """
    match = topk_idx.unsqueeze(-1) == pos_padded.unsqueeze(1)  # [B, K, P]
    p = pos_padded.size(1)
    pos_mask = torch.arange(p, device=topk_idx.device).unsqueeze(0) < pos_counts.unsqueeze(1)
    match = match & pos_mask.unsqueeze(1)
    return match.any(dim=-1).float()


def recall_at_k(topk_idx: torch.Tensor, pos_padded: torch.Tensor, pos_counts: torch.Tensor) -> torch.Tensor:
    """Per-user recall@K. Returns [B] float tensor."""
    hits = _build_hit_matrix(topk_idx, pos_padded, pos_counts)
    denom = pos_counts.clamp(min=1).to(hits.dtype)
    return hits.sum(dim=-1) / denom


def ndcg_at_k(topk_idx: torch.Tensor, pos_padded: torch.Tensor, pos_counts: torch.Tensor) -> torch.Tensor:
    """Per-user nDCG@K matching the reference ``ndcg_k``/``dcg_k`` definition."""
    hits = _build_hit_matrix(topk_idx, pos_padded, pos_counts)
    K = topk_idx.size(1)
    discount = 1.0 / torch.log2(
        torch.arange(2, K + 2, device=topk_idx.device, dtype=hits.dtype)
    )
    dcg = (hits * discount.unsqueeze(0)).sum(dim=-1)

    ideal_len = pos_counts.clamp(max=K).to(hits.dtype)
    ideal_mask = (
        torch.arange(K, device=topk_idx.device, dtype=hits.dtype).unsqueeze(0)
        < ideal_len.unsqueeze(1)
    ).to(hits.dtype)
    idcg = (ideal_mask * discount.unsqueeze(0)).sum(dim=-1)
    return dcg / idcg.clamp(min=1e-12)
