import time

import torch
from tqdm import tqdm


def compress_ppr_to_topk(ppr: torch.Tensor, topk: int, score_dtype: torch.dtype = torch.float16):
    """
    Compress dense user-node PPR scores to per-user top-k lists.

    Args:
        ppr:   [n_users, n_nodes] dense CPU tensor
        topk:  number of non-zero scores to retain per user
        score_dtype: dtype for cached scores

    Returns:
        dict with:
            mode:    'topk'
            indices: [n_users, topk] long CPU tensor
            scores:  [n_users, topk] CPU tensor
    """
    if ppr.dim() != 2:
        raise ValueError(f"Expected a 2D PPR tensor, got shape {tuple(ppr.shape)}")

    n_users, n_nodes = ppr.shape
    k = min(max(int(topk), 1), n_nodes)
    scores, indices = torch.topk(ppr.float(), k=k, dim=1)
    return {
        "mode": "topk",
        "indices": indices.to(dtype=torch.int32).cpu().contiguous(),
        "scores": scores.to(dtype=score_dtype).cpu().contiguous(),
        "ppr_topk": int(k),
    }



def _normalize_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device('cpu')

    dev = str(device).lower()
    if dev in {'auto', 'cuda_if_available'}:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA requested for PPR, but CUDA is unavailable. Falling back to CPU.')
        return torch.device('cpu')
    return torch.device(device)



def _run_ppr_once(
    loader,
    bs: int,
    N: int,
    alpha: float,
    beta: float,
    tol: float,
    topk: int | None,
    compute_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    device: torch.device,
):
    n_nodes = int(loader.n_nodes)
    n_users = int(loader.n_users)

    tkg = torch.as_tensor(loader.tKG, dtype=torch.long, device=device)
    heads = tkg[:, 0]
    tails = tkg[:, 2]

    # Preserve the original transition convention from the existing code.
    out_deg = torch.bincount(heads, minlength=n_nodes).clamp_min(1)
    values = (1.0 / out_deg[tails]).to(dtype=compute_dtype)
    indices = torch.stack([heads, tails], dim=0)
    M = torch.sparse_coo_tensor(
        indices, values, (n_nodes, n_nodes), dtype=compute_dtype, device=device
    ).coalesce()

    keep_topk = topk is not None and int(topk) > 0 and int(topk) < n_nodes
    if keep_topk:
        k = int(topk)
        final_indices = torch.empty((n_users, k), dtype=torch.int32)
        final_scores = torch.empty((n_users, k), dtype=cache_dtype)
    else:
        final_rank = torch.empty((n_users, n_nodes), dtype=cache_dtype)

    n_batch = n_users // bs + int(n_users % bs > 0)
    s_time = time.time()

    for i in tqdm(range(n_batch), desc=f'Computing PPR on {device.type}'):
        start = i * bs
        end = min(n_users, start + bs)
        tbs = end - start
        u_list = list(range(start, end))

        rank = torch.zeros((n_nodes, tbs), dtype=compute_dtype, device=device)
        user_idx = torch.tensor(u_list, dtype=torch.long, device=device)
        col_idx = torch.arange(tbs, dtype=torch.long, device=device)
        rank[user_idx, col_idx] = 1.0

        P = torch.empty((n_nodes, tbs), dtype=compute_dtype, device=device)
        for col, uid in enumerate(u_list):
            p_set = loader.known_user_set[uid]
            n_pref = len(p_set)

            if n_pref >= n_nodes:
                P[:, col].fill_(1.0 / n_nodes)
                continue

            denom = max(1, n_nodes - n_pref)
            base = (1.0 - beta) / denom
            P[:, col].fill_(base)

            if n_pref > 0:
                pref_idx = torch.as_tensor(list(p_set), dtype=torch.long, device=device)
                P[pref_idx, col] = beta / n_pref

        for _ in range(N):
            next_rank = (1.0 - alpha) * P + alpha * torch.sparse.mm(M, rank)
            if torch.norm(next_rank - rank).item() < tol:
                rank = next_rank
                break
            rank = next_rank

        rank_t = rank.transpose(0, 1).cpu()  # [tbs, n_nodes]
        if keep_topk:
            scores, idx = torch.topk(rank_t.float(), k=k, dim=1)
            final_indices[start:end] = idx.to(dtype=torch.int32)
            final_scores[start:end] = scores.to(dtype=cache_dtype)
        else:
            final_rank[start:end] = rank_t.to(dtype=cache_dtype)

        if device.type == 'cuda':
            del rank, P, rank_t
            torch.cuda.empty_cache()

    print(f'PPR done on {device.type}. time: {time.time() - s_time:.1f}s')

    if keep_topk:
        return {
            'mode': 'topk',
            'indices': final_indices.contiguous(),
            'scores': final_scores.contiguous(),
            'ppr_topk': int(k),
        }
    return final_rank.contiguous()



def get_ppr(
    loader,
    bs: int = 128,
    N: int = 20,
    alpha: float = 0.85,
    beta: float = 0.8,
    tol: float = 1e-6,
    topk: int | None = None,
    compute_dtype: torch.dtype = torch.float32,
    cache_dtype: torch.dtype = torch.float16,
    device: str | torch.device = 'auto',
    fallback_to_cpu: bool = True,
):
    """
    Compute Personalized PageRank (PPR) scores for each user over all nodes.

    Behavior:
    - Use CUDA for preprocessing when requested and available.
    - Move each finished user-batch back to CPU immediately.
    - If CUDA preprocessing fails due to memory pressure, optionally retry on CPU.

    Returns:
        Either:
            - dense CPU tensor [n_users, n_nodes] if topk is None / <= 0
            - dict(mode='topk', indices=[n_users, topk], scores=[n_users, topk])
    """
    compute_device = _normalize_device(device)

    try:
        return _run_ppr_once(
            loader=loader,
            bs=bs,
            N=N,
            alpha=alpha,
            beta=beta,
            tol=tol,
            topk=topk,
            compute_dtype=compute_dtype,
            cache_dtype=cache_dtype,
            device=compute_device,
        )
    except RuntimeError as e:
        msg = str(e).lower()
        oom_like = any(token in msg for token in ['out of memory', 'cuda error', 'cublas', 'cusparse'])
        if compute_device.type == 'cuda' and fallback_to_cpu and oom_like:
            print(f'GPU PPR preprocessing failed ({e}). Falling back to CPU once.')
            torch.cuda.empty_cache()
            return _run_ppr_once(
                loader=loader,
                bs=bs,
                N=N,
                alpha=alpha,
                beta=beta,
                tol=tol,
                topk=topk,
                compute_dtype=compute_dtype,
                cache_dtype=cache_dtype,
                device=torch.device('cpu'),
            )
        raise
