import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, d_model: int, max_period: float = 1e4):
        super().__init__()
        assert d_model % 2 == 0
        half = d_model // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] or [B,1]
        if t.dim() == 2 and t.size(1) == 1:
            t = t[:, 0]
        t = t.to(self.freqs.dtype)
        angles = t[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class NodeTimeNonCausalTransformer(nn.Module):
    """
    输入：一个事件 (src, dst, t) + 上下文事件窗口 tokens（包含past+future）
    输出：p_{src,t}, p_{dst,t}, p_{neg,t} （节点-时间社区分布）
    """
    def __init__(
        self,
        num_nodes: int,
        num_comms: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        learnable_time: bool = False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_comms = num_comms
        self.d_model = d_model

        self.node_emb = nn.Embedding(num_nodes, d_model)

        if learnable_time:
            self.time_enc = None
            self.time_proj = nn.Sequential(
                nn.Linear(1, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.time_enc = SinusoidalTimeEncoding(d_model)
            self.time_proj = nn.Identity()

        # event token: (u,v,t) -> d_model
        self.event_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # query token: (u,t) -> d_model
        self.query_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)

        # node-time -> community logits
        self.comm_head = nn.Linear(d_model, num_comms)

    def _t_embed(self, t: torch.Tensor) -> torch.Tensor:
        if self.time_enc is not None:
            return self.time_enc(t)
        return self.time_proj(t[:, None].to(torch.float32))

    def forward(
        self,
        src: torch.Tensor,          # [B]
        dst: torch.Tensor,          # [B]
        t: torch.Tensor,            # [B] float
        neg: torch.Tensor,          # [B,R]
        cu: torch.Tensor,           # [B,L] context edge src
        cv: torch.Tensor,           # [B,L] context edge dst
        ct: torch.Tensor,           # [B,L] context edge time float
        cmask: torch.Tensor,        # [B,L] bool, True=pad
    ):
        """
        返回：
          p_src: [B,K]
          p_dst: [B,K]
          p_neg: [B,R,K]
        """
        device = src.device
        B, L = cu.shape
        R = neg.shape[1]

        # ---- build context event tokens ----
        eu = self.node_emb(cu)  # [B,L,d]
        ev = self.node_emb(cv)  # [B,L,d]
        et = self._t_embed(ct.reshape(-1)).reshape(B, L, self.d_model)  # [B,L,d]
        ctx_tok = self.event_mlp(torch.cat([eu, ev, et], dim=-1))       # [B,L,d]
        ctx_tok = self.drop(ctx_tok)

        # ---- build query tokens for node-time: (src,t), (dst,t), (neg,t) ----
        t_emb = self._t_embed(t)  # [B,d]

        q_src = self.query_mlp(torch.cat([self.node_emb(src), t_emb], dim=-1))  # [B,d]
        q_dst = self.query_mlp(torch.cat([self.node_emb(dst), t_emb], dim=-1))  # [B,d]

        neg_flat = neg.reshape(-1)                     # [B*R]
        t_rep = t_emb.unsqueeze(1).expand(B, R, self.d_model).reshape(-1, self.d_model)  # [B*R,d]
        q_neg = self.query_mlp(torch.cat([self.node_emb(neg_flat), t_rep], dim=-1))      # [B*R,d]
        q_neg = q_neg.reshape(B, R, self.d_model)      # [B,R,d]

        # pack queries: [B, Q, d], Q = 2 + R
        q = torch.cat([q_src.unsqueeze(1), q_dst.unsqueeze(1), q_neg], dim=1)  # [B,2+R,d]
        q = self.drop(q)

        # transformer input: [queries | context]
        x = torch.cat([q, ctx_tok], dim=1)  # [B, 2+R+L, d]

        # padding mask: queries never padded
        qmask = torch.zeros(B, 2 + R, dtype=torch.bool, device=device)
        kpm = torch.cat([qmask, cmask], dim=1)  # [B, 2+R+L]

        # non-causal: no attn_mask (full bidirectional over provided tokens)
        h = self.encoder(x, src_key_padding_mask=kpm)  # [B, 2+R+L, d]
        hq = h[:, : 2 + R, :]                          # [B,2+R,d]

        h_src = hq[:, 0, :]        # [B,d]
        h_dst = hq[:, 1, :]        # [B,d]
        h_neg = hq[:, 2:, :]       # [B,R,d]

        logits_src = self.comm_head(h_src)             # [B,K]
        logits_dst = self.comm_head(h_dst)             # [B,K]
        logits_neg = self.comm_head(h_neg)             # [B,R,K]

        p_src = F.softmax(logits_src, dim=-1)
        p_dst = F.softmax(logits_dst, dim=-1)
        p_neg = F.softmax(logits_neg, dim=-1)

        return p_src, p_dst, p_neg, {"logits_src": logits_src, "logits_dst": logits_dst, "logits_neg": logits_neg}