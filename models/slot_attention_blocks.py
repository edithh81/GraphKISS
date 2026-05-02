import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttentionCore(nn.Module):
    """
    Slot Attention core module.

    Maps a set of N input vectors to K slot vectors through iterative competitive
    assignment. Attention is normalized over slots, not over inputs.
    """
    def __init__(
        self,
        dim_input: int,
        dim_slot: int,
        num_slots: int,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int | None = None,
        use_input_ln: bool = True,
        use_slot_ln: bool = True,
        use_mlp_ln: bool = True,
        use_gru: bool = True,
        use_residual_mlp: bool = True,
    ):
        super().__init__()
        if iters <= 0:
            raise ValueError(f"iters must be positive, got {iters}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}.")

        self.dim_input = dim_input
        self.dim_slot = dim_slot
        self.num_slots = num_slots
        self.iters = int(iters)
        self.eps = float(eps)
        self.use_gru = bool(use_gru)
        self.use_residual_mlp = bool(use_residual_mlp)

        hidden_dim = hidden_dim or (2 * dim_slot)

        self.norm_inputs = nn.RMSNorm(dim_input) if use_input_ln else nn.Identity()
        self.norm_slots = nn.RMSNorm(dim_slot) if use_slot_ln else nn.Identity()
        self.norm_mlp = nn.RMSNorm(dim_slot) if use_mlp_ln else nn.Identity()

        self.to_k = nn.Linear(dim_input, dim_slot, bias=False)
        self.to_v = nn.Linear(dim_input, dim_slot, bias=False)
        self.to_q = nn.Linear(dim_slot, dim_slot, bias=False)

        if self.use_gru:
            self.gru = nn.GRUCell(dim_slot, dim_slot)

        if self.use_residual_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(dim_slot, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim_slot),
            )

    def forward(
        self,
        inputs: torch.Tensor,
        slots: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [B, N, D_in]
            slots:  [B, K, D_slot]
            mask:   [B, N] bool, True at valid positions. If given, padded
                    positions are excluded from the weighted-mean aggregation.

        Returns:
            final_slots: [B, K, D_slot]
            attn:        [B, N, K] from the final iteration, normalized over slots
        """
        if inputs.dim() != 3:
            raise ValueError(f"inputs must have shape [B, N, D], got {tuple(inputs.shape)}.")
        if slots.dim() != 3:
            raise ValueError(f"slots must have shape [B, K, D], got {tuple(slots.shape)}.")
        if inputs.size(0) != slots.size(0):
            raise ValueError(
                f"Batch mismatch between inputs ({inputs.size(0)}) and slots ({slots.size(0)})."
            )
        if slots.size(1) != self.num_slots:
            raise ValueError(
                f"slots second dimension must equal num_slots={self.num_slots}, got {slots.size(1)}."
            )
        if slots.size(2) != self.dim_slot:
            raise ValueError(
                f"slots feature dim must equal dim_slot={self.dim_slot}, got {slots.size(2)}."
            )
        if mask is not None and mask.shape != inputs.shape[:2]:
            raise ValueError(
                f"mask must have shape [B, N]={tuple(inputs.shape[:2])}, got {tuple(mask.shape)}."
            )

        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)  # [B, N, D]
        v = self.to_v(inputs)  # [B, N, D]
        scale = 1.0 / math.sqrt(self.dim_slot)

        mask_f = mask.unsqueeze(-1).to(dtype=k.dtype) if mask is not None else None

        attn = None
        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)  # [B, K, D]

            logits = torch.einsum("bnd,bkd->bnk", k, q) * scale
            attn = F.softmax(logits, dim=-1)  # normalize over slots

            # Weighted mean aggregation; pad rows contribute 0 via the mask.
            attn_eps = attn + self.eps
            if mask_f is not None:
                attn_eps = attn_eps * mask_f
            denom = attn_eps.sum(dim=1, keepdim=True).clamp(min=self.eps)  # [B, 1, K]
            weights = attn_eps / denom
            updates = torch.einsum("bnk,bnd->bkd", weights, v)  # [B, K, D]

            if self.use_gru:
                slots = self.gru(
                    updates.reshape(-1, self.dim_slot),
                    slots_prev.reshape(-1, self.dim_slot),
                ).view_as(slots_prev)
            else:
                slots = updates

            if self.use_residual_mlp:
                slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn


class SlotAttentionIntentExtractor(nn.Module):
    """
    Tuple multiset -> user-specific intent slots using Slot Attention.

    Interface intentionally matches SetTransformerIntentExtractor so it can replace
    the old module with minimal changes to the rest of the codebase.
    """
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        num_outputs: int,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int | None = None,
        tuple_dropout: float = 0.0,
        learned_slot_init: bool = True,
        use_input_ln: bool = True,
        use_slot_ln: bool = True,
        use_mlp_ln: bool = True,
        use_gru: bool = True,
        use_residual_mlp: bool = True,
    ):
        super().__init__()
        if not (0.0 <= tuple_dropout < 1.0):
            raise ValueError(f"tuple_dropout must be in [0, 1), got {tuple_dropout}.")

        self.dim_hidden = dim_hidden
        self.num_outputs = num_outputs
        self.tuple_dropout = float(tuple_dropout)
        # True: stochastic Gaussian init (mu + eps * sigma), False: deterministic mu.
        self.learned_slot_init = bool(learned_slot_init)

        self.input_proj = (
            nn.Identity() if dim_input == dim_hidden else nn.Linear(dim_input, dim_hidden)
        )

        self.core = SlotAttentionCore(
            dim_input=dim_hidden,
            dim_slot=dim_hidden,
            num_slots=num_outputs,
            iters=iters,
            eps=eps,
            hidden_dim=hidden_dim,
            use_input_ln=use_input_ln,
            use_slot_ln=use_slot_ln,
            use_mlp_ln=use_mlp_ln,
            use_gru=use_gru,
            use_residual_mlp=use_residual_mlp,
        )

        # Per-slot orthogonal means: each slot starts in a different direction.
        self.slot_mu = nn.Parameter(torch.empty(1, num_outputs, dim_hidden))
        self._orthogonal_init_slots(self.slot_mu, gain=dim_hidden ** -0.5)

        # Per-slot log-sigma so each slot can learn its own spread (was shared across K).
        self.slot_log_sigma = nn.Parameter(
            torch.full((1, num_outputs, dim_hidden), -2.0)
        )

        # default_slots: also orthogonal, and decoupled from slot_mu.
        self.default_slots = nn.Parameter(torch.empty(1, num_outputs, dim_hidden))
        self._orthogonal_init_slots(self.default_slots, gain=dim_hidden ** -0.5)
    @staticmethod
    def _orthogonal_init_slots(param: nn.Parameter, gain: float = 1.0) -> None:
        """
        Orthogonal init for [1, K, D]. Handles K > D by tiling random orthogonal
        blocks (nn.init.orthogonal_ requires rows <= cols up to PyTorch's impl;
        this works for either shape).
        """
        K, D = param.shape[1], param.shape[2]
        with torch.no_grad():
            if K <= D:
                # K rows in R^D — standard orthogonal rows.
                w = torch.empty(K, D)
                nn.init.orthogonal_(w, gain=gain)
            else:
                # More slots than dims: stack ceil(K/D) orthogonal D x D blocks and slice.
                blocks = []
                for _ in range((K + D - 1) // D):
                    b = torch.empty(D, D)
                    nn.init.orthogonal_(b, gain=gain)
                    blocks.append(b)
                w = torch.cat(blocks, dim=0)[:K]
            param.copy_(w.unsqueeze(0))
            
    def get_default_slots(self, batch_size: int) -> torch.Tensor:
        return self.default_slots.repeat(batch_size, 1, 1)

    def _init_slots(self, batch_size, device, dtype):
        mu = self.slot_mu.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        if self.learned_slot_init:
            return mu
        # was: .expand(batch_size, self.num_outputs, -1) — now slot_log_sigma is already [1, K, D]
        sigma = self.slot_log_sigma.exp().expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        noise = torch.randn_like(mu)
        return mu + sigma * noise

    def forward(
        self,
        tuple_embeds: torch.Tensor,
        tuple_batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if tuple_embeds is None or tuple_embeds.numel() == 0:
            return self.get_default_slots(batch_size)

        device = tuple_embeds.device
        dtype = tuple_embeds.dtype
        N, D_in = tuple_embeds.shape

        counts = torch.bincount(tuple_batch, minlength=batch_size)  # [B]
        max_n = int(counts.max().item())
        if max_n == 0:
            return self.get_default_slots(batch_size)

        offsets = counts.cumsum(dim=0) - counts  # [B]
        pos_in_batch = torch.arange(N, device=device) - offsets[tuple_batch]  # [N]

        padded = tuple_embeds.new_zeros(batch_size, max_n, D_in)
        padded[tuple_batch, pos_in_batch] = tuple_embeds

        valid_mask = torch.zeros(batch_size, max_n, dtype=torch.bool, device=device)
        valid_mask[tuple_batch, pos_in_batch] = True

        if self.training and self.tuple_dropout > 0.0:
            keep_prob = 1.0 - self.tuple_dropout
            drop_mask = torch.rand(batch_size, max_n, device=device) < keep_prob
            combined = valid_mask & drop_mask
            any_kept = combined.any(dim=1)  # [B]
            had_tuples = counts > 0
            revert = had_tuples & ~any_kept  # rows where dropout killed every valid tuple
            combined = torch.where(revert.unsqueeze(1), valid_mask, combined)
            core_mask = combined
        else:
            core_mask = valid_mask

        x = self.input_proj(padded)
        slots0 = self._init_slots(batch_size=batch_size, device=device, dtype=dtype)
        slots, _ = self.core(inputs=x, slots=slots0, mask=core_mask)

        # Users with no tuples at all fall back to the learned default.
        empty_rows = counts == 0
        if empty_rows.any():
            default = self.default_slots.expand(batch_size, -1, -1)
            slots = torch.where(empty_rows.view(batch_size, 1, 1), default, slots)

        return slots
