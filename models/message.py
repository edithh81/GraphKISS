import torch
import torch.nn as nn

class KUCNetMessageFunction(nn.Module):
    """
    User-conditioned additive-attentive message function

        m_{s->o} = alpha_{sro} * W_msg(h_s + h_r)

    where
        alpha_{sro} = sigmoid( w_alpha^T ReLU(Ws h_s + Wr h_r + Wu h_u + b) )
    """
    def __init__(self, node_dim, attn_dim):
        super().__init__()
        self.node_dim = node_dim
        self.attn_dim = attn_dim

        self.Ws_attn = nn.Linear(node_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(node_dim, attn_dim, bias=True)
        self.w_alpha = nn.Linear(attn_dim, 1, bias=True)

        self.W_msg = nn.Linear(node_dim, node_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, h_src, rel_emb):
        """
        Args:
            h_src:        [E, D]
            rel_emb:      [E, D]
            h_user_edge:  [E, D]   user representation copied to each edge

        Returns:
            message:      [E, D]
            alpha:        [E]
        """
        base = h_src + rel_emb
        attn_feat = (
            self.Ws_attn(h_src)
            + self.Wr_attn(rel_emb)
        )
        alpha = torch.sigmoid(self.w_alpha(self.relu(attn_feat))).squeeze(-1)

        message = self.W_msg(base) * alpha.unsqueeze(-1)
        return message

class GRUMessageFunction(nn.Module):
    """
    GRU-based message function:
      m_{s→z}^ℓ = h_s^ℓ = GRU^{(ℓ)}(r_ℓ, h_s^{ℓ-1})

    - One GRUCell per layer.
    - The message dimension == node_dim.
    """
    def __init__(self, node_dim, rel_dim):
        super().__init__()
        self.node_dim = node_dim
        self.rel_dim = rel_dim

        self.gru = nn.GRUCell(rel_dim, node_dim)

    def forward(self, h_src, rel_emb):
        """
        Args:
            h_src:   [E, node_dim]  source node states h_s^{ℓ-1}
            rel_emb: [E, rel_dim]   relation embeddings r_ℓ for each edge
            layer_id: int, current GNN layer index

        Returns:
            messages: [E, node_dim] messages m_{s→z}^ℓ (updated hidden states)
        """
        messages = self.gru(rel_emb, h_src)   # [E, node_dim]
        return messages