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
        n_heads=4,
        n_layers=2,
        ff_mult=4,
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

        # time encoder
        if time_dim is None:
            time_dim = self.n_node_features
        self.time_encoder = TimeEncode(dimension=int(time_dim))

        # base node embedding from raw features (or learned table if disabled)
        self.init_from_node_features = bool(init_from_node_features)
        if self.init_from_node_features:
            self.node_lin = nn.Linear(self.n_node_features, self.embedding_dim)
        else:
            # fallback: learnable embedding table
            self.node_emb = nn.Embedding(self.n_nodes, self.embedding_dim)

        # edge feature -> embedding_dim
        self.edge_lin = nn.Linear(self.n_edge_features, self.embedding_dim)

        # occurrence token builder
        # token = [node_base, other_base, edge_emb, time_enc] -> proj to embedding_dim
        self.token_in_dim = self.embedding_dim * 3 + self.time_encoder.dimension
        self.token_lin = nn.Sequential(
            nn.Linear(self.token_in_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # bidirectional Transformer over per-node sequences
        # Use batch_first=True for easier padding: [B_nodes, L, D]
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=int(n_heads),
            dim_feedforward=int(ff_mult) * self.embedding_dim,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))

        # community projector
        self.community_projector = CommunityProjector(
            embedding_dim=self.embedding_dim,
            num_communities=self.num_communities,
            dropout=dropout,
        )

        # store t0 just for stable absolute-time encoding if needed
        self.t0 = float(t0)

    def _node_base(self, nodes: torch.Tensor) -> torch.Tensor:
        """Return base embedding for nodes: [*, D]."""
        if self.init_from_node_features:
            return self.node_lin(self.node_raw_features[nodes])
        return self.node_emb(nodes)

    def _time_enc_abs(self, ts: torch.Tensor) -> torch.Tensor:
        """Absolute time encoding (ts - t0). ts: [*] float -> [*, T]."""
        dt = (ts - float(self.t0)).to(dtype=torch.float32)
        return self.time_encoder(dt.view(-1, 1)).view(*ts.shape, -1)

    def _build_occurrence_tokens(self, src: torch.Tensor, dst: torch.Tensor, ts: torch.Tensor, eidx: torch.Tensor):
        """\
        Build per-event occurrence tokens.

        Returns:
          occ_nodes: LongTensor [2B]  (node id for each occurrence)
          occ_times: FloatTensor[2B]
          occ_tok:   FloatTensor[2B, D]  (token embedding BEFORE transformer)
          occ_meta:  dict with mapping helpers
        """
        B = src.numel()

        # occurrences: first B are (src @ t) with other=dst, next B are (dst @ t) with other=src
        occ_nodes = torch.cat([src, dst], dim=0)              # [2B]
        other_nodes = torch.cat([dst, src], dim=0)            # [2B]
        occ_times = torch.cat([ts, ts], dim=0)                # [2B]
        occ_eidx = torch.cat([eidx, eidx], dim=0)             # [2B]

        node_base = self._node_base(occ_nodes)                # [2B, D]
        other_base = self._node_base(other_nodes)             # [2B, D]
        edge_emb = self.edge_lin(self.edge_raw_features[occ_eidx])  # [2B, D]
        time_emb = self._time_enc_abs(occ_times)              # [2B, T]

        x = torch.cat([node_base, other_base, edge_emb, time_emb], dim=-1)  # [2B, 3D+T]
        tok = self.token_lin(x)                                # [2B, D]

        meta = {
            "B": B,
            "occ_is_src": torch.cat([
                torch.ones(B, device=self.device, dtype=torch.bool),
                torch.zeros(B, device=self.device, dtype=torch.bool)
            ], dim=0),
        }
        return occ_nodes, occ_times, tok, meta

    def _encode_per_node_sequences(self, occ_nodes: torch.Tensor, occ_times: torch.Tensor, occ_tok: torch.Tensor):
        """\
        Group occurrences by node, sort by time, pad to max length, run Transformer.

        Returns:
          node_ids: LongTensor [Nn]
          out_pad:  FloatTensor [Nn, Lmax, D]
          pos_map:  LongTensor [2B]  position index within the node sequence for each occurrence
          node_row: LongTensor [2B]  row index in [0..Nn-1] for each occurrence
        """
        # unique nodes appearing in occurrences
        node_ids = torch.unique(occ_nodes)
        Nn = node_ids.numel()

        # map node id -> row index
        # (use scatter-friendly mapping via a dict on CPU for simplicity; Nn is small per batch)
        node_ids_cpu = node_ids.detach().cpu().tolist()
        row_of = {int(n): i for i, n in enumerate(node_ids_cpu)}

        # collect indices per node
        occ_nodes_cpu = occ_nodes.detach().cpu().tolist()
        buckets = [[] for _ in range(Nn)]
        for j, n in enumerate(occ_nodes_cpu):
            buckets[row_of[int(n)]].append(j)

        # sort each node's occurrences by time
        # and record position mapping
        pos_map = torch.empty((occ_nodes.numel(),), device=self.device, dtype=torch.long)
        node_row = torch.empty((occ_nodes.numel(),), device=self.device, dtype=torch.long)

        seq_lens = []
        sorted_idx_per_row = []
        for r in range(Nn):
            idxs = buckets[r]
            if len(idxs) == 0:
                seq_lens.append(0)
                sorted_idx_per_row.append([])
                continue
            tvals = occ_times[idxs]
            order = torch.argsort(tvals)
            idxs_sorted = [idxs[int(k)] for k in order.detach().cpu().tolist()]
            sorted_idx_per_row.append(idxs_sorted)
            seq_lens.append(len(idxs_sorted))
            for p, j in enumerate(idxs_sorted):
                pos_map[j] = p
                node_row[j] = r

        Lmax = int(max(seq_lens)) if len(seq_lens) > 0 else 0
        if Lmax == 0:
            # no occurrences
            out_pad = occ_tok.new_zeros((Nn, 1, self.embedding_dim))
            return node_ids, out_pad, pos_map, node_row

        # pad sequences
        x_pad = occ_tok.new_zeros((Nn, Lmax, self.embedding_dim))
        key_padding = torch.ones((Nn, Lmax), device=self.device, dtype=torch.bool)  # True=pad
        for r in range(Nn):
            idxs_sorted = sorted_idx_per_row[r]
            L = len(idxs_sorted)
            if L == 0:
                continue
            x_pad[r, :L, :] = occ_tok[idxs_sorted]
            key_padding[r, :L] = False

        # transformer encoding
        out_pad = self.transformer(x_pad, src_key_padding_mask=key_padding)  # [Nn, Lmax, D]
        return node_ids, out_pad, pos_map, node_row

    def compute_community_prob(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs):
        """\
        Inputs:
          source_nodes, destination_nodes: array-like [B]
          negative_nodes: array-like [B] or [B,R]
          edge_times: array-like [B]
          edge_idxs: array-like [B]

        Returns:
          p_src: [B,K]
          p_dst: [B,K]
          p_neg: [B,K] or [B,R,K]
        """
        src = torch.as_tensor(source_nodes, dtype=torch.long, device=self.device)
        dst = torch.as_tensor(destination_nodes, dtype=torch.long, device=self.device)
        ts = torch.as_tensor(edge_times, dtype=torch.float32, device=self.device)
        eidx = torch.as_tensor(edge_idxs, dtype=torch.long, device=self.device)

        B = src.numel()
        if B == 0:
            # return empty tensors with correct shapes
            p_src = torch.empty((0, self.num_communities), device=self.device)
            p_dst = torch.empty((0, self.num_communities), device=self.device)
            p_neg = torch.empty((0, self.num_communities), device=self.device)
            return p_src, p_dst, p_neg

        # Build occurrence tokens (2B tokens)
        occ_nodes, occ_times, occ_tok, _ = self._build_occurrence_tokens(src, dst, ts, eidx)

        # ---- Negatives: encode with the SAME token structure + Transformer ----
        # Build pseudo-occurrence tokens for negatives at the same (t, edge) with other=src.
        neg = torch.as_tensor(negative_nodes, dtype=torch.long, device=self.device)

        if neg.dim() == 1:
            assert neg.numel() == B
            R = 1
            neg_flat = neg
            ts_rep = ts
            src_rep = src
            eidx_rep = eidx
        elif neg.dim() == 2:
            assert neg.shape[0] == B
            R = int(neg.shape[1])
            neg_flat = neg.reshape(-1)                                   # [B*R]
            ts_rep = ts.unsqueeze(1).expand(B, R).reshape(-1)             # [B*R]
            src_rep = src.unsqueeze(1).expand(B, R).reshape(-1)           # [B*R]
            eidx_rep = eidx.unsqueeze(1).expand(B, R).reshape(-1)         # [B*R]
        else:
            raise ValueError(f"negative_nodes must be 1D or 2D, got shape={tuple(neg.shape)}")

        # token for negative occurrences: (neg_node @ t) with other=src, edge=eidx
        neg_base = self._node_base(neg_flat)                              # [B*R, D]
        other_base = self._node_base(src_rep)                             # [B*R, D]
        edge_emb = self.edge_lin(self.edge_raw_features[eidx_rep])        # [B*R, D]
        time_emb = self._time_enc_abs(ts_rep)                             # [B*R, T]
        x_neg = torch.cat([neg_base, other_base, edge_emb, time_emb], dim=-1)
        neg_tok = self.token_lin(x_neg)                                   # [B*R, D]

        # concat occurrences (positives first, then negatives)
        pos_occ = occ_nodes.numel()                                       # 2B
        occ_nodes_all = torch.cat([occ_nodes, neg_flat], dim=0)
        occ_times_all = torch.cat([occ_times, ts_rep], dim=0)
        occ_tok_all = torch.cat([occ_tok, neg_tok], dim=0)

        # encode all occurrences with the same per-node Transformer
        _, out_pad, pos_map, node_row = self._encode_per_node_sequences(occ_nodes_all, occ_times_all, occ_tok_all)

        # gather src/dst
        src_occ = torch.arange(0, B, device=self.device)
        dst_occ = torch.arange(B, 2 * B, device=self.device)
        src_h = out_pad[node_row[src_occ], pos_map[src_occ], :]           # [B, D]
        dst_h = out_pad[node_row[dst_occ], pos_map[dst_occ], :]           # [B, D]
        p_src = self.community_projector(src_h)
        p_dst = self.community_projector(dst_h)

        # gather negatives
        neg_start = pos_occ
        neg_idx = torch.arange(neg_start, neg_start + neg_flat.numel(), device=self.device)
        neg_h = out_pad[node_row[neg_idx], pos_map[neg_idx], :]           # [B*R, D]
        if R == 1:
            p_neg = self.community_projector(neg_h)                       # [B, K]
        else:
            p_neg = self.community_projector(neg_h).view(B, R, -1)        # [B, R, K]

        return p_src, p_dst, p_neg