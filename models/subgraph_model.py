import os

import torch
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F

from .message import GRUMessageFunction, KUCNetMessageFunction
from .aggregator import (
    FullPNAAggregator,
    SimplifiedPNAAggregator,
    KUCNetAggregator,
    DegreeScalerSumAggregator,
)
from .scorer import FiLMIntentConditioner, IntentSlotNodeScorer, LowRankIntentItemScorer
from .slot_attention_blocks import SlotAttentionIntentExtractor
from ppr import compress_ppr_to_topk, get_ppr

INTERACT_REL_ID = 0
INTERACT_INV_REL_ID = 1


class GraphNorm(nn.Module):
    """
    GraphNorm with learnable mean-subtraction weight alpha.
        y = gamma * (x - alpha * mean_graph) / std_graph + beta
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.alpha = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x, batch):
        count = scatter(torch.ones(x.size(0), 1, device=x.device), batch, dim=0, reduce="sum").clamp(min=1)
        mean = scatter(x, batch, dim=0, reduce="sum") / count
        x_centered = x - self.alpha * mean[batch]
        var = scatter(x_centered ** 2, batch, dim=0, reduce="sum") / count
        std = (var[batch] + self.eps).sqrt()
        return self.gamma * x_centered / std + self.beta


def variadic_topk_edges(probs, counts, k, device):
    """
    Top-k edge selection per variable-length group, vectorized via pad + topk.

    probs  : [E]     edge scores, pre-sorted so each group's rows are contiguous
    counts : [G]     per-group sizes (sum(counts) == E)
    k      : int     max keep per group
    Returns global edge indices (into `probs`) of the kept rows.
    """
    if probs.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    counts = counts.to(device=device, dtype=torch.long)
    num_groups = counts.numel()
    max_count = int(counts.max().item())
    actual_k = min(k, max_count)

    # Scatter flat probs into a [G, max_count] padded matrix (padded with -inf).
    offsets = counts.cumsum(dim=0) - counts                         # [G]
    group_idx = torch.repeat_interleave(
        torch.arange(num_groups, device=device), counts
    )                                                               # [E]
    pos_in_group = torch.arange(probs.numel(), device=device) - offsets[group_idx]  # [E]

    padded = probs.new_full((num_groups, max_count), float("-inf"))
    padded[group_idx, pos_in_group] = probs

    _, local_idx = padded.topk(actual_k, dim=1)                     # [G, actual_k]

    # Drop pad positions where local_idx >= counts[g] (only hits when k>size).
    valid = local_idx < counts.unsqueeze(1)                         # [G, actual_k]
    global_idx = local_idx + offsets.unsqueeze(1)                   # [G, actual_k]
    return global_idx[valid]

def _dcor(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Distance correlation between two [n, d] matrices.

    dCor(X, Y) = dCov(X, Y) / sqrt(dVar(X) * dVar(Y))

    Pairwise Euclidean distance matrices are double-centred; dCov² is the
    mean element-wise product of the two centred matrices.
    """
    n = X.size(0)

    def _dc(M):
        row = M.mean(dim=1, keepdim=True)
        col = M.mean(dim=0, keepdim=True)
        return M - row - col + M.mean()

    Ax = _dc(torch.cdist(X, X, p=2))   # [n, n]
    Ay = _dc(torch.cdist(Y, Y, p=2))   # [n, n]

    dcov2_xy = (Ax * Ay).sum() / (n * n)
    dvar_x   = (Ax * Ax).sum() / (n * n)
    dvar_y   = (Ay * Ay).sum() / (n * n)

    return (dcov2_xy / (dvar_x * dvar_y + eps).sqrt()).clamp(min=0.0)


def intent_diversity_loss(intent_slots: torch.Tensor) -> torch.Tensor:
    """
    Intra-user distance-correlation independence regularizer.

    Treats the batch dimension as samples and computes dCor between every
    pair of slot columns across all users:
        L_intra = Σ_{1≤k<k'≤K} dCor(intent_slots[:,k,:], intent_slots[:,k',:])

    intent_slots is assumed L2-normalised before this call.
    """
    if intent_slots is None or intent_slots.numel() == 0:
        device = intent_slots.device if intent_slots is not None else "cpu"
        return torch.tensor(0.0, device=device)
    if intent_slots.dim() != 3:
        raise ValueError(
            f"intent_slots must have shape [B, K, D], got {tuple(intent_slots.shape)}."
        )
    K = intent_slots.size(1)
    if K <= 1:
        return intent_slots.new_tensor(0.0)

    loss = intent_slots.new_tensor(0.0)
    for k in range(K):
        for kp in range(k + 1, K):
            loss = loss + _dcor(intent_slots[:, k, :], intent_slots[:, kp, :])
    return loss


def centroid_orthogonality_loss(intent_slots: torch.Tensor) -> torch.Tensor:
    """
    Cross-user distance-correlation independence regularizer.

    Computes batch-mean centroids c_k = mean_u p_{u,k} ∈ R^D, then measures
    pairwise dCor between centroids treating the D feature dimensions as samples:
        L_cross = Σ_{1≤k<k'≤K} dCor(c_k, c_k')

    Prevents cross-user collapse: all users' slot k drifting to the same direction.
    """
    if intent_slots is None or intent_slots.numel() == 0:
        device = intent_slots.device if intent_slots is not None else "cpu"
        return torch.tensor(0.0, device=device)
    if intent_slots.dim() != 3:
        raise ValueError(
            f"intent_slots must have shape [B, K, D], got {tuple(intent_slots.shape)}."
        )
    K = intent_slots.size(1)
    if K <= 1:
        return intent_slots.new_tensor(0.0)

    centroids = intent_slots.mean(dim=0)   # [K, D]
    loss = intent_slots.new_tensor(0.0)
    for k in range(K):
        for kp in range(k + 1, K):
            # treat D dimensions as n samples, each centroid value as a scalar obs
            loss = loss + _dcor(centroids[k].unsqueeze(-1), centroids[kp].unsqueeze(-1))
    return loss


def score_edges_with_batch_ppr(edge_batch, tail_nodes, batch_ppr, device):
    """
    Score candidate edges by gathering from a batch-local dense PPR scratch.

    ``batch_ppr["values"]`` is a ``[B, n_nodes]`` tensor built once per batch in
    ``AdaptiveSubgraphModel._build_batch_ppr`` — either directly when the model
    caches dense PPR, or scattered from the top-k cache otherwise. Non-cached
    tails are zero, so lookup becomes one gather of cost O(E) instead of the
    O(E * ppr_topk) match loop the previous implementation ran.
    """
    if batch_ppr is None:
        raise ValueError("batch_ppr is None while PPR scoring was requested.")

    values = batch_ppr["values"]
    if edge_batch.numel() == 0:
        return torch.zeros(0, dtype=values.dtype, device=device)
    return values[edge_batch, tail_nodes]


def sample_topk_with_gumbel(logits: torch.Tensor, k: int, tau_g: float, training: bool):
    """Straight-through Gumbel-TopK sampler for one candidate set."""
    if logits.numel() == 0:
        return logits.new_zeros((0,), dtype=torch.long), logits.new_zeros((0,))

    if tau_g <= 0:
        raise ValueError(f"tau_g must be positive, got {tau_g}.")

    actual_k = min(k, logits.numel())
    if training:
        u = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-10)
        gumbel = -torch.log(-torch.log(u))
        perturbed = logits + gumbel
        relaxed_probs = torch.softmax(perturbed / tau_g, dim=0)
        _, selected_local_idx = torch.topk(perturbed, k=actual_k)
    else:
        relaxed_probs = torch.softmax(logits / tau_g, dim=0)
        _, selected_local_idx = torch.topk(logits, k=actual_k)

    return selected_local_idx, relaxed_probs


class TupleEmbedder(nn.Module):
    """
    Tuple encoder for hop-2 candidate edges.

    For each valid hop-2 edge (item --r--> tail), build a tuple embedding using
    concatenation followed by an MLP:
        x_(r,t) = MLP([z_r || h_t])

    No deduplication and no frequency truncation are applied. Repeated tuples remain
    repeated elements in the tuple multiset.
    """
    def __init__(self, node_dim: int, n_rel: int, n_user: int, n_item: int):
        super().__init__()
        self.node_dim = node_dim
        self.n_rel = n_rel
        self.n_user = n_user
        self.n_item = n_item

        self.tuple_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.LeakyReLU(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(self, edges, h_tilde, rela_embed):
        self_loop_id = self.n_rel * 2 + 2

        rel_id = edges[:, 2].long()
        head_idx = edges[:, 1].long()
        tail_global_idx = edges[:, 3].long()
        tail_local_idx = edges[:, 5].long()
        batch_idx = edges[:, 0].long()

        valid = (
            (rel_id != self_loop_id)
            & (rel_id != INTERACT_REL_ID)
            & (rel_id != INTERACT_INV_REL_ID)
            & (head_idx >= self.n_user)
            & (head_idx < self.n_user + self.n_item)
        )

        if not valid.any():
            empty_long = edges.new_zeros((0,), dtype=torch.long)
            empty_emb = h_tilde.new_zeros((0, self.node_dim))
            return {
                "batch_idx": empty_long,
                "rel_id": empty_long,
                "head_id": empty_long,
                "tail_id": empty_long,
                "tuple_emb": empty_emb,
            }

        rel_vec = rela_embed(rel_id[valid])
        tail_vec = h_tilde[tail_local_idx[valid]]
        tuple_emb = self.tuple_mlp(torch.cat([rel_vec, tail_vec], dim=-1))

        return {
            "batch_idx": batch_idx[valid],
            "rel_id": rel_id[valid],
            "head_id": head_idx[valid],
            "tail_id": tail_global_idx[valid],
            "tuple_emb": tuple_emb,
        }


class AdaptiveSubgraphLayer(nn.Module):
    """
    One message-passing layer for user-centric subgraph reasoning.

    Intent is not injected into message passing. It is only used for node scoring.
    """
    def __init__(
        self,
        node_dim,
        n_user,
        n_item,
        n_rel,
        n_node,
        PNA_delta=None,
        K=50,
        K_ppr=0,
        K_intent=4,
        tau_s=1.0,
        tau_g=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        item_bonus=None,
        message_type="gru",
        aggregator_type="pna_full",
        attn_dim=None,
        use_graph_norm=True,
    ):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_rel = n_rel
        self.n_node = n_node
        self.node_dim = node_dim
        self.K = K
        self.K_ppr = K_ppr
        self.K_intent = K_intent
        self.tau_g = float(tau_g)
        self.device = device
        self.item_bonus = item_bonus
        self.use_graph_norm = bool(use_graph_norm)
        self.ppr_debug = False

        self.rela_embed = nn.Embedding(2 * n_rel + 1 + 2, node_dim)

        if message_type == "gru":
            self.message_fn = GRUMessageFunction(node_dim=node_dim, rel_dim=node_dim)
        elif message_type == "kucnet":
            self.message_fn = KUCNetMessageFunction(
                node_dim=node_dim,
                attn_dim=attn_dim if attn_dim is not None else node_dim,
            )
        else:
            raise ValueError(f"Unknown message_type: {message_type!r}")

        if aggregator_type == "pna_full":
            self.aggregator = FullPNAAggregator(node_dim=node_dim, delta=PNA_delta)
            self._agg_uses_h_prev = True
        elif aggregator_type == "pna_simple":
            self.aggregator = SimplifiedPNAAggregator(node_dim=node_dim)
            self._agg_uses_h_prev = True
        elif aggregator_type == "degree_sum":
            self.aggregator = DegreeScalerSumAggregator(node_dim=node_dim, delta=PNA_delta)
            self._agg_uses_h_prev = True
        elif aggregator_type == "kucnet":
            self.aggregator = KUCNetAggregator(node_dim=node_dim)
            self._agg_uses_h_prev = False
        else:
            raise ValueError(f"Unknown aggregator_type: {aggregator_type!r}")

        self.graph_norm = GraphNorm(node_dim)
        self.scorer = IntentSlotNodeScorer(node_dim=node_dim, K_intent=K_intent, tau=tau_s)

    def _compute_h_user(self, q_sub, nodes, h_tilde):
        batch_size = q_sub.size(0)
        _, dim = h_tilde.shape
        node_batch = nodes[:, 0].long()
        node_ent = nodes[:, 1].long()

        target_ent = q_sub[node_batch]
        mask = node_ent == target_ent

        h_user = torch.zeros(batch_size, dim, device=self.device)
        h_user.index_copy_(0, node_batch[mask], h_tilde[mask])
        return h_user

    def forward(
    self,
    q_sub,
    q_rel,
    hidden,
    edges,
    nodes,
    id_layer,
    n_layer,
    old_nodes_new_idx,
    intent_state=None,
    tuple_embedder=None,
    intent_extractor=None,
    batch_ppr=None,
    film_conditioner=None,
    ):
        del q_rel  # unused; kept in signature for compatibility

        # ---- Edge PPR pruning (unchanged) ----
        if batch_ppr is not None and self.K_ppr > 0 and 0 < id_layer < n_layer - 1:
            _, sort_order = torch.unique(edges[:, 0:3], dim=0, sorted=True, return_inverse=True)
            _, sort_idx = torch.sort(sort_order)
            edges = edges[sort_idx]

            _, _, group_counts = torch.unique(
                edges[:, 0:3], dim=0, return_inverse=True, return_counts=True
            )

            edge_batch = edges[:, 0].long()
            tail_nodes = edges[:, 3].long()
            probs = score_edges_with_batch_ppr(edge_batch, tail_nodes, batch_ppr, self.device)

            keep_idx = variadic_topk_edges(probs, group_counts, self.K_ppr, self.device)

            if self.ppr_debug:
                kept_probs = probs[keep_idx]
                total = kept_probs.numel()
                if total > 0:
                    in_cache = int((kept_probs > 0).sum().item())
                    frac = 100.0 * in_cache / total
                    print(
                        f"[PPR-debug L{id_layer}] kept={total} in_cache={in_cache} "
                        f"({frac:.1f}%) zero_fill={total - in_cache}"
                    )

            edges = edges[keep_idx]

            nodes, tail_index = torch.unique(edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
            edges = torch.cat([edges[:, 0:5], tail_index.unsqueeze(1)], dim=1)

            idd_mask = edges[:, 2] == (self.n_rel * 2 + 2)
            _, old_sort = edges[:, 4][idd_mask].sort()
            old_nodes_new_idx = tail_index[idd_mask][old_sort]

        # ---- Last-layer item-tail filter ----
        # The last hop has no PPR pruning and no K-sampling, so it would MP
        # every reachable hop-3 edge and only drop non-items at the very end
        # via `keep_mask &= is_item`. That MP work on non-item destinations is
        # wasted: those h_tilde rows are computed and immediately discarded.
        # Keep only edges whose tail is an item or the query user (the user is
        # needed so its self-loop survives for `_compute_h_user`).
        last_layer_filtered = False
        if id_layer == n_layer - 1:
            tail_global = edges[:, 3]
            edge_batch = edges[:, 0]
            is_item_tail = (
                (tail_global >= self.n_user)
                & (tail_global < self.n_user + self.n_item)
            )
            is_query_user_tail = tail_global == q_sub[edge_batch]
            keep_edge = is_item_tail | is_query_user_tail
            edges = edges[keep_edge]

            nodes, tail_index = torch.unique(
                edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True
            )
            edges = torch.cat([edges[:, 0:5], tail_index.unsqueeze(1)], dim=1)

            self_loop_id = self.n_rel * 2 + 2
            idd_mask = edges[:, 2] == self_loop_id
            # Partial mapping: prev-frontier -> new-frontier via surviving self-loops.
            # Used below to gather hidden rows into h_prev_new without assuming
            # contiguous prev-frontier coverage.
            _filtered_prev_idx = edges[idd_mask, 4]
            _filtered_new_idx = tail_index[idd_mask]
            old_nodes_new_idx = _filtered_new_idx
            last_layer_filtered = True

        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]

        # NOTE: _last_num_messages is set at the END of forward, after node
        # sampling and (last-layer) item filtering, so it reflects messages
        # whose destination node actually survives all pruning stages.

        hs = hidden[sub]
        hr = self.rela_embed(rel)
        messages = self.message_fn(h_src=hs, rel_emb=hr)

        n_nodes = nodes.size(0)
        dim = hidden.size(1)
        h_prev_new = torch.zeros(n_nodes, dim, device=self.device)
        if last_layer_filtered:
            # `old_nodes_new_idx` is no longer aligned with hidden's row order
            # because non-item / non-user prev nodes were dropped. Rebuild via
            # the (prev_idx, new_idx) pairs from surviving self-loops.
            h_prev_new[_filtered_new_idx] = hidden[_filtered_prev_idx]
        else:
            h_prev_new.index_copy_(0, old_nodes_new_idx, hidden)

        if self._agg_uses_h_prev:
            h_tilde = self.aggregator(messages=messages, dst_index=obj, h_prev=h_prev_new)
        else:
            h_tilde = self.aggregator(messages=messages, dst_index=obj, num_nodes=n_nodes)

        batch_size = q_sub.size(0)
        node_batch = nodes[:, 0].long()
        if self.use_graph_norm:
            h_tilde = self.graph_norm(h_tilde, node_batch)
        h_user = self._compute_h_user(q_sub, nodes, h_tilde)

        # Build intent slots from hop-2 candidate set before hop-2 pruning.
        if id_layer == 1 and intent_state is None and tuple_embedder is not None and intent_extractor is not None:
            tuple_records = tuple_embedder(edges=edges, h_tilde=h_tilde, rela_embed=self.rela_embed)

            intent_slots = intent_extractor(
                tuple_embeds=tuple_records["tuple_emb"],
                tuple_batch=tuple_records["batch_idx"],
                batch_size=batch_size,
            )

            intent_slots = F.normalize(intent_slots, p=2, dim=-1)

            intent_state = {
                "tuple_embeds": tuple_records["tuple_emb"],
                "tuple_batch": tuple_records["batch_idx"],
                "tuple_rel_ids": tuple_records["rel_id"],
                "intent_slots": intent_slots,
                "intent_div_loss": intent_diversity_loss(intent_slots),
                "centroid_ortho_loss": centroid_orthogonality_loss(intent_slots),
            }

        # Layer 1 (hop-1 items) is fully kept. Sampling starts from hop 2.
        if id_layer == 0:
            node_score = torch.zeros(h_tilde.size(0), device=self.device)
            hidden_all = h_tilde
        else:
            if intent_state is None or "intent_slots" not in intent_state:
                raise ValueError("intent_state with intent_slots is required from hop 2 onward.")
            if film_conditioner is None:
                raise ValueError("film_conditioner is required from hop 2 onward.")
            user_views = film_conditioner(
                h_user=h_user,
                intent_slots=intent_state["intent_slots"],
            )
            node_score, node_score_k = self.scorer(
                user_views=user_views,
                h_node=h_tilde,
                node_batch=node_batch,
            )
            hidden_all = h_tilde

        keep_mask = torch.zeros(n_nodes, dtype=torch.bool, device=self.device)
        keep_mask[old_nodes_new_idx] = True

        diff_mask = torch.ones(n_nodes, dtype=torch.bool, device=self.device)
        diff_mask[old_nodes_new_idx] = False
        candidate_idx = diff_mask.nonzero(as_tuple=False).squeeze(-1)

        if self.K is not None and self.K > 0 and candidate_idx.numel() > 0 and id_layer != 0 and id_layer < n_layer - 1:
            cand_batch = node_batch[candidate_idx]
            cand_score = node_score[candidate_idx]
            if self.item_bonus is not None:
                cand_node_ids = nodes[candidate_idx, 1]
                cand_is_item = (
                    (cand_node_ids >= self.n_user)
                    & (cand_node_ids < self.n_user + self.n_item)
                )
                cand_score = cand_score + cand_is_item.float() * self.item_bonus

            # Sort candidates by batch so each batch's candidates are contiguous.
            sort_order = torch.argsort(cand_batch, stable=True)
            sorted_idx = candidate_idx[sort_order]
            sorted_batch = cand_batch[sort_order]
            sorted_score = cand_score[sort_order]
            Nc = sorted_score.numel()

            counts = torch.bincount(sorted_batch, minlength=batch_size)
            max_count = int(counts.max().item())
            actual_k = min(self.K, max_count)

            offsets = counts.cumsum(dim=0) - counts
            pos_in_batch = torch.arange(Nc, device=self.device) - offsets[sorted_batch]

            padded_score = cand_score.new_full(
                (batch_size, max_count), float("-inf")
            )
            padded_score[sorted_batch, pos_in_batch] = sorted_score

            if self.training:
                u = torch.rand_like(padded_score).clamp_(1e-10, 1.0 - 1e-10)
                gumbel = -torch.log(-torch.log(u))
                perturbed = padded_score + gumbel  # pad rows stay -inf
                relaxed_probs = torch.softmax(perturbed / self.tau_g, dim=-1)
                _, top_local = torch.topk(perturbed, actual_k, dim=-1)
            else:
                _, top_local = torch.topk(padded_score, actual_k, dim=-1)
                relaxed_probs = None

            valid = top_local < counts.unsqueeze(1)  # [B, actual_k]
            safe_local = top_local.clamp(max=max(max_count - 1, 0))
            sorted_global = offsets.unsqueeze(1) + safe_local  # [B, actual_k]
            sorted_global = sorted_global.clamp(max=max(Nc - 1, 0))
            selected_flat = sorted_idx[sorted_global][valid]
            keep_mask[selected_flat] = True

            if self.training and selected_flat.numel() > 0:
                probs_selected = relaxed_probs.gather(1, safe_local)
                st_coeff = 1.0 - probs_selected.detach() + probs_selected
                st_coeff = st_coeff[valid]

                # ST coefficient as a [N, 1] multiplier (1 elsewhere). Avoids
                # cloning the full [N, D] hidden tensor for autograd.
                mult = hidden_all.new_ones(n_nodes, 1)
                mult = mult.scatter(0, selected_flat.unsqueeze(-1), st_coeff.unsqueeze(-1))
                hidden_all = hidden_all * mult
        else:
            keep_mask[:] = True

        if id_layer == n_layer - 1:
            hidden_user = self._compute_h_user(q_sub, nodes, hidden_all)
            is_item = (nodes[:, 1] >= self.n_user) & (nodes[:, 1] < self.n_user + self.n_item)
            keep_mask &= is_item
            final_nodes = nodes[keep_mask]
        else:
            final_nodes = None
            hidden_user = None

        sampled_nodes_idx = keep_mask.clone()
        hidden_new = hidden_all[keep_mask]
        nodes_new = nodes[keep_mask]

        # ---- Effective message count: messages whose destination node survives
        # edge PPR pruning + node sampling (+ item filtering on last layer). ----
        self._last_num_messages = int(keep_mask[obj].sum().item())

        return (
            hidden_new,
            nodes_new,
            final_nodes,
            hidden_user,
            old_nodes_new_idx,
            sampled_nodes_idx,
            node_score,
            edges,
            intent_state,
        )


class AdaptiveSubgraphModel(torch.nn.Module):
    def __init__(self, params, loader, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.n_rel = params.n_rel
        self.n_users = params.n_users
        self.n_items = params.n_items
        self.n_nodes = params.n_nodes
        self.loader = loader
        self.device = device

        PNA_delta = getattr(params, "PNA_delta", None)
        K = getattr(params, "K", 50)
        K_ppr = getattr(params, "K_ppr", 0)
        self.K_intent = getattr(params, "K_intent", 4)
        ppr_topk = getattr(params, "ppr_topk", 0)
        ppr_compute_bs = getattr(params, "ppr_compute_bs", 128)
        ppr_device = getattr(params, "ppr_device", "auto")
        ppr_fallback_to_cpu = getattr(params, "ppr_fallback_to_cpu", True)
        ppr_debug = bool(getattr(params, "ppr_debug", False))
        item_bonus_init = float(getattr(params, "item_bonus_init", 0.05))

        message_type = str(getattr(params, "message_type", "gru"))
        aggregator_type = str(getattr(params, "aggregator_type", "pna_full"))
        attn_dim = getattr(params, "attn_dim", None)
        use_graph_norm = bool(getattr(params, "use_graph_norm", True))

        self.edge_counts_layer: list[int] = []

        self.tau_s = float(getattr(params, "tau_s", getattr(params, "tau", 1.0)))
        self.tau_p = float(getattr(params, "tau_p", getattr(params, "tau", 1.0)))
        self.tau_g = float(getattr(params, "tau_g", 1.0))
        self.pred_rank = int(getattr(params, "pred_rank", max(8, self.hidden_dim // 2)))

        slot_attn_iters = int(getattr(params, "slot_attn_iters", 3))
        slot_attn_eps = float(getattr(params, "slot_attn_eps", 1e-8))
        slot_attn_mlp_hidden = int(getattr(params, "slot_attn_mlp_hidden", 2 * self.hidden_dim))
        slot_attn_use_input_ln = bool(getattr(params, "slot_attn_use_input_ln", True))
        slot_attn_use_slot_ln = bool(getattr(params, "slot_attn_use_slot_ln", True))
        slot_attn_use_mlp_ln = bool(getattr(params, "slot_attn_use_mlp_ln", True))
        slot_attn_use_gru = bool(getattr(params, "slot_attn_use_gru", True))
        slot_attn_use_residual_mlp = bool(getattr(params, "slot_attn_use_residual_mlp", True))
        slot_attn_learned_slot_init = bool(getattr(params, "slot_attn_learned_slot_init", True))
        tuple_dropout = float(getattr(params, "tuple_dropout", 0.0))

        self.item_bonus = nn.Parameter(
            torch.tensor(item_bonus_init, dtype=torch.float32)
        )

        ppr_cache_dir = getattr(params, "ppr_cache_dir", os.path.join("results", "ppr_cache"))
        data_tag = os.path.basename(os.path.normpath(getattr(loader, "task_dir", "dataset")))
        ppr_mode_tag = f"top{ppr_topk}" if ppr_topk and ppr_topk > 0 else "dense"
        ppr_cache_file = os.path.join(ppr_cache_dir, f"{data_tag}_ppr_{ppr_mode_tag}.pt")

        self.ppr_store = None
        self.ppr_topk = int(ppr_topk) if ppr_topk is not None else 0

        if K_ppr > 0:
            os.makedirs(ppr_cache_dir, exist_ok=True)
            payload = None

            if os.path.exists(ppr_cache_file):
                try:
                    payload = torch.load(ppr_cache_file, map_location="cpu", weights_only=True)
                except Exception as e:
                    print(f"Safe load failed ({e}). Trying trusted fallback once...")
                    try:
                        payload = torch.load(ppr_cache_file, map_location="cpu", weights_only=False)
                    except Exception as e2:
                        print(f"Failed to load cached PPR ({e2}). Recomputing PPR.")

            def _meta_ok(obj):
                return (
                    int(obj.get("n_users", self.n_users)) == int(self.n_users)
                    and int(obj.get("n_nodes", self.n_nodes)) == int(self.n_nodes)
                    and int(obj.get("n_rel", self.n_rel)) == int(self.n_rel)
                    and int(obj.get("n_items", self.n_items)) == int(self.n_items)
                )

            if isinstance(payload, dict) and _meta_ok(payload):
                mode = payload.get("mode", None)
                if mode == "topk":
                    idx = payload.get("indices", None)
                    scores = payload.get("scores", None)
                    if (
                        torch.is_tensor(idx)
                        and torch.is_tensor(scores)
                        and idx.size(0) == self.n_users
                        and scores.shape == idx.shape
                    ):
                        self.ppr_store = {
                            "mode": "topk",
                            "indices": idx.to(dtype=torch.int32).cpu().contiguous(),
                            "scores": scores.cpu().contiguous(),
                            "ppr_topk": int(idx.size(1)),
                        }
                        print(f"Loaded cached top-k PPR from: {ppr_cache_file}")
                elif mode == "dense":
                    cached_ppr = payload.get("ppr", None)
                    if torch.is_tensor(cached_ppr) and cached_ppr.shape == (self.n_users, self.n_nodes):
                        dense_cpu = cached_ppr.cpu().contiguous()
                        if self.ppr_topk > 0:
                            self.ppr_store = compress_ppr_to_topk(dense_cpu, self.ppr_topk, score_dtype=torch.float16)
                            print(f"Loaded cached dense PPR and compressed to top-{self.ppr_topk} at model init.")
                        else:
                            self.ppr_store = {"mode": "dense", "values": dense_cpu}
                            print(f"Loaded cached dense PPR from: {ppr_cache_file}")
                elif "ppr" in payload:
                    cached_ppr = payload.get("ppr", None)
                    if torch.is_tensor(cached_ppr) and cached_ppr.shape == (self.n_users, self.n_nodes):
                        dense_cpu = cached_ppr.cpu().contiguous()
                        if self.ppr_topk > 0:
                            self.ppr_store = compress_ppr_to_topk(dense_cpu, self.ppr_topk, score_dtype=torch.float16)
                            print(f"Loaded legacy dense PPR cache and compressed to top-{self.ppr_topk}.")
                        else:
                            self.ppr_store = {"mode": "dense", "values": dense_cpu.to(dtype=torch.float16)}
                            print(f"Loaded legacy dense PPR cache from: {ppr_cache_file}")
            elif torch.is_tensor(payload) and payload.shape == (self.n_users, self.n_nodes):
                dense_cpu = payload.cpu().contiguous()
                if self.ppr_topk > 0:
                    self.ppr_store = compress_ppr_to_topk(dense_cpu, self.ppr_topk, score_dtype=torch.float16)
                    print(f"Loaded legacy tensor PPR and compressed to top-{self.ppr_topk}.")
                else:
                    self.ppr_store = {"mode": "dense", "values": dense_cpu.to(dtype=torch.float16)}
                    print(f"Loaded legacy dense tensor PPR from: {ppr_cache_file}")

            if self.ppr_store is None:
                print(f"Computing PPR scores (precompute device={ppr_device}, CPU cache, model-level storage)...")
                fresh_ppr = get_ppr(
                    loader,
                    bs=ppr_compute_bs,
                    topk=self.ppr_topk if self.ppr_topk > 0 else None,
                    cache_dtype=torch.float16,
                    device=ppr_device,
                    fallback_to_cpu=ppr_fallback_to_cpu,
                )
                if isinstance(fresh_ppr, dict):
                    self.ppr_store = {
                        "mode": "topk",
                        "indices": fresh_ppr["indices"].to(dtype=torch.int32).cpu().contiguous(),
                        "scores": fresh_ppr["scores"].cpu().contiguous(),
                        "ppr_topk": int(fresh_ppr["indices"].size(1)),
                    }
                else:
                    dense_cpu = fresh_ppr.cpu().contiguous()
                    self.ppr_store = {"mode": "dense", "values": dense_cpu}

                try:
                    if self.ppr_store["mode"] == "topk":
                        torch.save(
                            {
                                "mode": "topk",
                                "indices": self.ppr_store["indices"],
                                "scores": self.ppr_store["scores"],
                                "ppr_topk": int(self.ppr_store["indices"].size(1)),
                                "n_users": int(self.n_users),
                                "n_nodes": int(self.n_nodes),
                                "n_rel": int(self.n_rel),
                                "n_items": int(self.n_items),
                                "task_dir": str(getattr(loader, "task_dir", "")),
                            },
                            ppr_cache_file,
                        )
                    else:
                        torch.save(
                            {
                                "mode": "dense",
                                "ppr": self.ppr_store["values"].cpu().contiguous(),
                                "n_users": int(self.n_users),
                                "n_nodes": int(self.n_nodes),
                                "n_rel": int(self.n_rel),
                                "n_items": int(self.n_items),
                                "task_dir": str(getattr(loader, "task_dir", "")),
                            },
                            ppr_cache_file,
                        )
                    print(f"Saved PPR cache to: {ppr_cache_file}")
                except Exception as e:
                    print(f"Warning: failed to save PPR cache ({e}).")

        self.tuple_embedder = TupleEmbedder(
            node_dim=self.hidden_dim,
            n_rel=self.n_rel,
            n_user=self.n_users,
            n_item=self.n_items,
        )
        self.intent_extractor = SlotAttentionIntentExtractor(
            dim_input=self.hidden_dim,
            dim_hidden=self.hidden_dim,
            num_outputs=self.K_intent,
            iters=slot_attn_iters,
            eps=slot_attn_eps,
            hidden_dim=slot_attn_mlp_hidden,
            tuple_dropout=tuple_dropout,
            learned_slot_init=slot_attn_learned_slot_init,
            use_input_ln=slot_attn_use_input_ln,
            use_slot_ln=slot_attn_use_slot_ln,
            use_mlp_ln=slot_attn_use_mlp_ln,
            use_gru=slot_attn_use_gru,
            use_residual_mlp=slot_attn_use_residual_mlp,
        )
        self.film_conditioner = FiLMIntentConditioner(node_dim=self.hidden_dim)
        self.pred_scorer = LowRankIntentItemScorer(
            node_dim=self.hidden_dim,
            rank=self.pred_rank,
        )

        self.gnn_layers = nn.ModuleList([
            AdaptiveSubgraphLayer(
                node_dim=self.hidden_dim,
                n_user=self.n_users,
                n_item=self.n_items,
                n_rel=self.n_rel,
                n_node=self.n_nodes,
                PNA_delta=PNA_delta,
                K=K,
                K_ppr=K_ppr,
                K_intent=self.K_intent,
                tau_s=self.tau_s,
                tau_g=self.tau_g,
                device=self.device,
                item_bonus=self.item_bonus,
                message_type=message_type,
                aggregator_type=aggregator_type,
                attn_dim=attn_dim,
                use_graph_norm=use_graph_norm,
            )
            for _ in range(self.n_layer)
        ])
        for layer in self.gnn_layers:
            layer.ppr_debug = ppr_debug

        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

    def _build_batch_ppr(self, q_sub):
        if self.ppr_store is None:
            return None

        user_ids = q_sub.detach().cpu()
        batch_size = user_ids.numel()
        mode = self.ppr_store.get("mode", None)
        if mode == "dense":
            values = self.ppr_store["values"].index_select(0, user_ids).to(
                self.device, dtype=torch.float32, non_blocking=True
            )
            return {"mode": "dense", "values": values}
        if mode == "topk":
            idx_cpu = self.ppr_store["indices"].index_select(0, user_ids)
            scores_cpu = self.ppr_store["scores"].index_select(0, user_ids)
            idx_gpu = idx_cpu.to(self.device, non_blocking=True).long()
            scores_gpu = scores_cpu.to(self.device, dtype=torch.float32, non_blocking=True)
            dense = torch.zeros(
                (batch_size, self.n_nodes), dtype=torch.float32, device=self.device
            )
            dense.scatter_(1, idx_gpu, scores_gpu)
            return {"mode": "dense", "values": dense}
        raise ValueError(f"Unsupported model-level PPR store mode: {mode}")

    def forward(self, subs, rels, mode="train", return_aux: bool = False):
        n = len(subs)

        q_sub = torch.LongTensor(subs).to(self.device)
        q_rel = torch.LongTensor(rels).to(self.device)

        nodes = torch.cat([torch.arange(n).unsqueeze(1).to(self.device), q_sub.unsqueeze(1)], dim=1)
        hidden = torch.zeros(n, self.hidden_dim).to(self.device)
        h0 = torch.zeros((1, n, self.hidden_dim)).to(self.device)

        final_nodes = None
        hidden_user = None
        intent_state = None
        batch_ppr = self._build_batch_ppr(q_sub)

        self.edge_counts_layer = []

        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors_gpu(nodes, mode=mode)

            tuple_embedder = self.tuple_embedder if i == 1 and intent_state is None else None
            intent_extractor = self.intent_extractor if i == 1 and intent_state is None else None

            (
                hidden,
                nodes,
                final_nodes,
                hidden_user,
                old_nodes_new_idx,
                sampled_nodes_idx,
                node_score,
                edges,
                intent_state,
            ) = self.gnn_layers[i](
                q_sub,
                q_rel,
                hidden,
                edges,
                nodes,
                i,
                self.n_layer,
                old_nodes_new_idx,
                intent_state=intent_state,
                tuple_embedder=tuple_embedder,
                intent_extractor=intent_extractor,
                batch_ppr=batch_ppr,
                film_conditioner=self.film_conditioner,
            )
            self.edge_counts_layer.append(getattr(self.gnn_layers[i], "_last_num_messages", 0))

            n_nodes_this_layer = sampled_nodes_idx.size(0)
            h0_full = torch.zeros(1, n_nodes_this_layer, hidden.size(1), device=self.device)
            # old_nodes_new_idx maps prev-frontier positions -> current-layer node positions.
            # h0 has shape [1, prev_frontier_size, D]. Only copy positions whose
            # current-layer node survived sampling (sampled_nodes_idx[pos] == True).
            survived_mask = sampled_nodes_idx[old_nodes_new_idx]  # [prev_frontier_size] bool
            prev_positions = torch.arange(old_nodes_new_idx.size(0), device=self.device)
            survived_new_idx = old_nodes_new_idx[survived_mask]   # current-layer positions that survived
            survived_prev_idx = prev_positions[survived_mask]     # corresponding h0 columns
            if survived_new_idx.numel() > 0:
                h0_full[:, survived_new_idx, :] = h0[:, survived_prev_idx, :]
            h0 = h0_full[:, sampled_nodes_idx, :]

            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

        if final_nodes is None:
            raise ValueError("final_nodes is None after forward pass.")

        if intent_state is None or "intent_slots" not in intent_state:
            intent_slots = self.intent_extractor.get_default_slots(n).to(self.device)
        else:
            intent_slots = intent_state["intent_slots"]

        item_batch = final_nodes[:, 0].long()
        h_i = hidden

        user_views = self.film_conditioner(
            h_user=hidden_user,
            intent_slots=intent_slots,
        )[item_batch]

        scores_k = self.pred_scorer(
            user_views=user_views,
            item_emb=h_i,
        )
        scores = torch.logsumexp(self.tau_p * scores_k, dim=-1) / self.tau_p

        scores_all = torch.zeros((n, self.n_items), device=self.device)
        scores_all[(final_nodes[:, 0], final_nodes[:, 1] - self.n_users)] = scores

        zero = torch.tensor(0.0, device=self.device)
        intent_div_loss = intent_state.get("intent_div_loss", zero) if intent_state is not None else zero
        centroid_ortho_loss = intent_state.get("centroid_ortho_loss", zero) if intent_state is not None else zero

        if return_aux:
            return scores_all, {
                "intent_div_loss": intent_div_loss,
                "centroid_ortho_loss": centroid_ortho_loss,
            }
        return scores_all
