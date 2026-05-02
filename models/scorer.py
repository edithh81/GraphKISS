import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMIntentConditioner(nn.Module):
    """
    Transform user-specific intent slots into FiLM parameters and produce
    intent-conditioned user views.

    For each user u and intent slot k:
        [gamma_{u,k}, beta_{u,k}] = MLP_film(i_{u,k})
        h_tilde_{u,k} = (1 + gamma_{u,k}) ⊙ RMSNorm(h_u) + beta_{u,k}

    Prenorm FiLM with RMSNorm: h_u is RMS-normalized before modulation, so
    gamma and beta operate on a clean unit-RMS input. Output scale is free —
    downstream temperatures (tau_s, tau_p) may need re-tuning.
    """
    def __init__(self, node_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.LeakyReLU(),
            nn.Linear(node_dim, 2 * node_dim),
        )
        self.norm = nn.RMSNorm(node_dim)

    def forward(self, h_user: torch.Tensor, intent_slots: torch.Tensor) -> torch.Tensor:
        if h_user.dim() != 2:
            raise ValueError(f"h_user must have shape [B, D], got {tuple(h_user.shape)}.")
        if intent_slots.dim() != 3:
            raise ValueError(f"intent_slots must have shape [B, K, D], got {tuple(intent_slots.shape)}.")

        batch_size, num_slots, dim = intent_slots.shape
        if h_user.size(0) != batch_size:
            raise ValueError(
                f"Batch mismatch between h_user ({h_user.size(0)}) and intent_slots ({batch_size})."
            )
        if h_user.size(1) != dim:
            raise ValueError(
                f"Feature mismatch between h_user ({h_user.size(1)}) and intent_slots ({dim})."
            )

        film_params = self.mlp(intent_slots.reshape(batch_size * num_slots, dim)).view(
            batch_size, num_slots, 2 * dim
        )
        gamma, beta = torch.chunk(film_params, chunks=2, dim=-1)

        h_u = self.norm(h_user).unsqueeze(1).expand(batch_size, num_slots, dim)
        return (1.0 + gamma) * h_u + beta


class IntentSlotNodeScorer(nn.Module):
    """
    FiLM-conditioned user-view node scorer.

    For node v belonging to query b:
        s_{b,v,k} = w2 . LeakyReLU(W1_slot @ h_tilde_{b,k} + W1_node @ h_v + b1)
        s_{b,v}   = (1 / tau) * log sum_k exp(tau * s_{b,v,k})

    The first linear of the MLP is factored as W1 = [W1_slot | W1_node] so we
    never materialize the [N, K, 2D] concatenation. Mathematically equivalent
    to the previous nn.Linear(2D, D) form.
    """
    def __init__(self, node_dim: int, K_intent: int | None = None, tau: float = 1.0):
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}.")

        self.K_intent = K_intent
        self.tau = float(tau)
        self.W1_slot = nn.Linear(node_dim, node_dim, bias=False)
        self.W1_node = nn.Linear(node_dim, node_dim, bias=True)
        self.W2 = nn.Linear(node_dim, 1)

    def forward(
        self,
        user_views: torch.Tensor,
        h_node: torch.Tensor,
        node_batch: torch.Tensor,
        intent_bias: torch.Tensor | None = None,
    ):
        if user_views.dim() != 3:
            raise ValueError(f"user_views must have shape [B, K, D], got {tuple(user_views.shape)}.")
        if h_node.dim() != 2:
            raise ValueError(f"h_node must have shape [N, D], got {tuple(h_node.shape)}.")
        if node_batch.dim() != 1:
            raise ValueError(f"node_batch must have shape [N], got {tuple(node_batch.shape)}.")

        # Project on the small [B, K, D] tensor first; index by node only after.
        slot_h = self.W1_slot(user_views)               # [B, K, D]
        node_h = self.W1_node(h_node)                   # [N, D]
        pre_act = slot_h[node_batch] + node_h.unsqueeze(1)  # [N, K, D]
        s_k = self.W2(F.leaky_relu(pre_act)).squeeze(-1)    # [N, K]

        if intent_bias is not None:
            k = s_k.size(-1)
            if intent_bias.numel() != k:
                raise ValueError(
                    f"intent_bias must have {k} entries, got {intent_bias.numel()}."
                )
            s_k = s_k + intent_bias.view(1, k)
        s_v = torch.logsumexp(self.tau * s_k, dim=-1) / self.tau
        return s_v, s_k


class LowRankIntentItemScorer(nn.Module):
    """
    Low-rank bilinear scorer for final recommendation.

    Replaces the plain dot product with
        s_{u,i,k} = h_tilde_{u,k}^T W h_{i|u},
    where W is parameterized in low-rank form via two learnable projections:
        W ≈ P^T Q,
        s_{u,i,k} = <P h_tilde_{u,k}, Q h_{i|u}>.

    This preserves the intended bilinear interaction while keeping parameter
    growth controlled.
    """
    def __init__(self, node_dim: int, rank: int):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")

        self.node_dim = node_dim
        self.rank = int(rank)
        self.user_proj = nn.Linear(node_dim, self.rank, bias=False)
        self.item_proj = nn.Linear(node_dim, self.rank, bias=False)

    def forward(
        self,
        user_views: torch.Tensor,
        item_emb: torch.Tensor,
        intent_bias: torch.Tensor | None = None,
    ):
        if user_views.dim() != 3:
            raise ValueError(f"user_views must have shape [M, K, D], got {tuple(user_views.shape)}.")
        if item_emb.dim() != 2:
            raise ValueError(f"item_emb must have shape [M, D], got {tuple(item_emb.shape)}.")
        if user_views.size(0) != item_emb.size(0):
            raise ValueError(
                f"Batch mismatch between user_views ({user_views.size(0)}) and item_emb ({item_emb.size(0)})."
            )
        if user_views.size(-1) != item_emb.size(-1):
            raise ValueError(
                f"Feature mismatch between user_views ({user_views.size(-1)}) and item_emb ({item_emb.size(-1)})."
            )

        m, k, d = user_views.shape
        user_latent = self.user_proj(user_views.reshape(m * k, d)).view(m, k, self.rank)
        item_latent = self.item_proj(item_emb).unsqueeze(1)
        scores_k = (user_latent * item_latent).sum(dim=-1)
        if intent_bias is not None:
            if intent_bias.numel() != k:
                raise ValueError(
                    f"intent_bias must have {k} entries, got {intent_bias.numel()}."
                )
            scores_k = scores_k + intent_bias.view(1, k)
        return scores_k
