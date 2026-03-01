import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tgn.model.time_encoding import TimeEncode


class CommunityProjector(nn.Module):
    def __init__(self, embedding_dim, num_communities, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_communities),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.mlp(x)


class CommunityModel(nn.Module):
    """
    Event-level community inference model (JODIE-style):
      - Each node has a state vector s[v] and last update time last_t[v].
      - At time t, we "project" state to current time using a decay gate.
      - Community prob p(v,t) = softmax(Projector(Proj(s[v], t-last_t[v]))).
      - Only endpoints (src,dst) are updated after processing real events.
      - Negative nodes are NOT updated; only used as contrasts in the loss.

    Interface:
      compute_community_prob(src, dst, neg_nodes, edge_times, edge_idxs)
      update_states_from_events(src, dst, edge_times, edge_idxs)  # call in training loop after optimizer.step()
    """

    def __init__(
        self,
        node_features,
        edge_features,
        device,
        *,
        embedding_dim=None,
        num_communities=5,
        dropout=0.1,
        time_dim=None,
        init_from_node_features=True,
        t0=0.0,
    ):
        super().__init__()
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

        self.n_nodes = self.node_raw_features.shape[0]
        self.n_node_features = self.node_raw_features.shape[1]
        self.n_edge_features = self.edge_raw_features.shape[1]

        self.embedding_dim = int(embedding_dim) if embedding_dim is not None else int(self.n_node_features)
        self.num_communities = int(num_communities)

        # time encoder (reuse your existing module)
        if time_dim is None:
            time_dim = self.n_node_features
        self.time_encoder = TimeEncode(dimension=int(time_dim))

        # ---- node state table (BUFFER, not a Parameter) ----
        # state[v] will be updated with torch.no_grad() in update_states_from_events
        self.register_buffer("state", torch.zeros(self.n_nodes, self.embedding_dim, device=device))
        self.register_buffer("last_t", torch.full((self.n_nodes,), float(t0), device=device))

        if init_from_node_features:
            # map node features -> embedding_dim as initial state
            self.init_lin = nn.Linear(self.n_node_features, self.embedding_dim)
        else:
            self.init_lin = None

        # ---- time projection: exponential decay gate ----
        # decay_rate = softplus(param) so it's positive
        self.log_decay = nn.Parameter(torch.tensor(0.0, device=device))  # learnable scalar

        # ---- event update (shared GRUCell) ----
        # input = [other_proj, edge_feat, time_enc(dt_self)]
        self.update_input_dim = self.embedding_dim + self.n_edge_features + self.time_encoder.dimension
        self.gru = nn.GRUCell(self.update_input_dim, self.embedding_dim)

        # community projector
        self.community_projector = CommunityProjector(
            embedding_dim=self.embedding_dim,
            num_communities=self.num_communities,
            dropout=dropout
        )

        # initialize state
        #self.reset_state(t0=float(t0))

    @torch.no_grad()
    def reset_state(self, t0=0.0):
        """Reset state table and last_t (useful per epoch if needed)."""
        self.last_t.fill_(float(t0))
        if self.init_lin is None:
            self.state.zero_()
        else:
            self.state.copy_(self.init_lin(self.node_raw_features))

    def _proj(self, s: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Project state forward in time by dt (dt>=0).
        Exponential decay: s_proj = s * exp(-decay * dt)
        """
        dt = torch.clamp(dt, min=0.0).to(dtype=s.dtype)
        decay = F.softplus(self.log_decay)  # positive scalar
        gate = torch.exp(-decay * dt).unsqueeze(-1)  # [...,1]
        return s * gate

    def _get_proj_state(self, nodes: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        nodes: LongTensor [...]
        t:     FloatTensor [...] (same shape)
        Returns:
          s_proj: Tensor [...,D]
          dt:     Tensor [...]  (t-last_t)
        """
        last = self.last_t[nodes].to(dtype=t.dtype)
        dt = t - last
        dt = torch.clamp(dt, min=0.0)
        s = self.state[nodes]
        s_proj = self._proj(s, dt)
        return s_proj, dt

    def compute_community_prob(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs):
        """
        Inputs:
          source_nodes, destination_nodes: array-like [B]
          negative_nodes: array-like [B] or [B,R]
          edge_times: array-like [B]
          edge_idxs: array-like [B]

        Returns:
          p_src: [B,K]
          p_dst: [B,K]
          p_neg: [B,K] or [B,R,K] (matches negative_nodes layout)
        """
        # to torch
        src = torch.as_tensor(source_nodes, dtype=torch.long, device=self.device)
        dst = torch.as_tensor(destination_nodes, dtype=torch.long, device=self.device)
        ts = torch.as_tensor(edge_times, dtype=torch.float32, device=self.device)

        B = src.numel()
        assert dst.numel() == B and ts.numel() == B

        # projected states for endpoints
        s_src, dt_src = self._get_proj_state(src, ts)
        s_dst, dt_dst = self._get_proj_state(dst, ts)

        # negative nodes: [B] or [B,R]
        neg = torch.as_tensor(negative_nodes, dtype=torch.long, device=self.device)
        if neg.dim() == 1:
            assert neg.numel() == B
            s_neg, _ = self._get_proj_state(neg, ts)  # [B,D]
        elif neg.dim() == 2:
            assert neg.shape[0] == B
            R = neg.shape[1]
            ts_rep = ts.unsqueeze(1).expand(B, R)          # [B,R]
            neg_flat = neg.reshape(-1)                      # [B*R]
            ts_flat = ts_rep.reshape(-1)                    # [B*R]
            s_neg_flat, _ = self._get_proj_state(neg_flat, ts_flat)  # [B*R,D]
            s_neg = s_neg_flat.view(B, R, -1)               # [B,R,D]
        else:
            raise ValueError(f"negative_nodes must be 1D or 2D, got shape={tuple(neg.shape)}")

        # community probs
        p_src = self.community_projector(s_src)  # [B,K]
        p_dst = self.community_projector(s_dst)  # [B,K]
        p_neg = self.community_projector(s_neg)  # [B,K] or [B,R,K]

        return p_src, p_dst, p_neg

    @torch.no_grad()
    def update_states_from_events(self, source_nodes, destination_nodes, edge_times, edge_idxs):
        """
        Update state/last_t for REAL events only (src & dst).
        Call this after optimizer.step() to avoid backprop through the state table.

        IMPORTANT: This assumes edge_times are in nondecreasing order overall.
        If not, we sort within the batch by time to avoid negative dt.
        """
        src = torch.as_tensor(source_nodes, dtype=torch.long, device=self.device)
        dst = torch.as_tensor(destination_nodes, dtype=torch.long, device=self.device)
        ts = torch.as_tensor(edge_times, dtype=torch.float32, device=self.device)
        eidx = torch.as_tensor(edge_idxs, dtype=torch.long, device=self.device)

        B = src.numel()
        if B == 0:
            return

        # sort within batch by time to avoid time going backwards for the same node
        order = torch.argsort(ts)
        src = src[order]
        dst = dst[order]
        ts = ts[order]
        eidx = eidx[order]

        # edge features for each event
        edge_feat = self.edge_raw_features[eidx]  # [B, E]

        # sequential update so duplicate nodes in a batch are handled correctly
        for i in range(B):
            u = src[i]
            v = dst[i]
            t = ts[i]
            ef = edge_feat[i].unsqueeze(0)  # [1,E]

            # project both endpoints to time t
            su, dtu = self._get_proj_state(u.view(1), t.view(1))  # [1,D], [1]
            sv, dtv = self._get_proj_state(v.view(1), t.view(1))  # [1,D], [1]

            tu_enc = self.time_encoder(dtu.view(1, 1)).view(1, -1)  # [1,T]
            tv_enc = self.time_encoder(dtv.view(1, 1)).view(1, -1)

            # update u using v as context
            inp_u = torch.cat([sv, ef, tu_enc], dim=1)  # [1, D+E+T]
            new_su = self.gru(inp_u, su)                # [1,D]

            # update v using u as context (use projected su, not updated su, to keep symmetry)
            inp_v = torch.cat([su, ef, tv_enc], dim=1)
            new_sv = self.gru(inp_v, sv)

            # write back
            self.state[u] = new_su.squeeze(0)
            self.state[v] = new_sv.squeeze(0)
            self.last_t[u] = t
            self.last_t[v] = t