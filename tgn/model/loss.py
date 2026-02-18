import torch
from typing import Dict, Tuple
import logging
import math


def longitudinal_modularity_loss(p_src, p_dst, src, dst, delta_a_nodes,
                                 A_base, S_base, node_lifespan, global_degree,m, total_duration, p_prev, delta_src,delta_dst,
                                 csc_norm="l2", obs_weight=1.0,
                                 null_weight=1.0, csc_weight=1.0):
    device = p_src.device
    B = p_src.size(0)
    # observation loss
    loss_obs = -(p_src * p_dst).sum(dim=1).mean()

    K = int(p_src.size(1))

    # We treat each endpoint at each interaction as an "occurrence" (time-node).
    # Build occurrence-level observed adjacency A_occ (only true interaction pairs),
    # and an occurrence-level expected term E_occ consistent with modularity: (A - E) structure.
    P_list = []   # list of p_o in R^K
    S_list = []   # list of s_o scalars (k_u * alpha_u)
    edge_occ_pairs = []  # list of (idx_src_occ, idx_dst_occ) per interaction

    def _append_occ(p_u: torch.Tensor, u: int, dt: torch.Tensor) -> int:
        dtv = dt.to(dtype=p_u.dtype)
        if dtv.item() <= 0:
            raise ValueError("Negative delta time.")

        ku = float(global_degree[u])
        Tu = float(node_lifespan[u])
        alpha = dtv / (Tu + 1.0)

        p_o = p_u
        s_o = ku * alpha

        idx = len(P_list)
        P_list.append(p_o)
        S_list.append(p_o.new_tensor(s_o))
        return idx

    for i in range(B):
        idx_u = _append_occ(p_src[i], int(src[i].item()), delta_src[i])
        idx_v = _append_occ(p_dst[i], int(dst[i].item()), delta_dst[i])
        edge_occ_pairs.append((idx_u, idx_v))

    M_occ = len(P_list)
    if M_occ <= 1:
        loss_null = torch.zeros((), device=device, dtype=p_src.dtype)
    else:
        P_occ = torch.stack(P_list, dim=0)                 # [M_occ, K]
        S_occ = torch.stack(S_list, dim=0)                 # [M_occ]

        # Similarity between occurrence community assignments: <p_o, p_o'>
        Sim = P_occ @ P_occ.t()                            # [M_occ, M_occ]

        # Observed adjacency at occurrence level: A_occ[o,o']=1 iff (o,o') is an observed interaction pair
        A_occ = torch.zeros((M_occ, M_occ), device=device, dtype=p_src.dtype)
        for (a, b) in edge_occ_pairs:
            A_occ[a, b] = A_occ[a, b] + 1.0
            A_occ[b, a] = A_occ[b, a] + 1.0

        # Modularity denominator (occurrence-graph): use 2mB (same scale as your previous normalization)
        eps = 1e-12
        denom = (2.0 * float(B)) + eps

        # Expected term at occurrence level: E_occ[o,o'] = (s_o * s_o') / denom
        E_occ = (S_occ[:, None] * S_occ[None, :]) /(2*m)   # [M_occ, M_occ]

        # Remove diagonal (optional but usually desirable): self-pairs shouldn't contribute.
        A_occ = A_occ - torch.diag(torch.diag(A_occ))
        E_occ = E_occ - torch.diag(torch.diag(E_occ))
        Sim = Sim - torch.diag(torch.diag(Sim))

        # Modularity-style objective: Q = (1/denom) * sum_{o,o'} (A_occ - E_occ) * <p_o, p_o'>
        Q = ((A_occ - E_occ) * Sim).sum() / denom

        # We MINIMIZE loss, so negate modularity objective
        loss_null = -Q
    """
    # null model loss
    eps = 1e-12
    A0 = A_base
    S0 = S_base
    S_corr = torch.zeros_like(S0)
    for u, dA_u in delta_a_nodes.items():
        ku = global_degree[u]
        base = A0[u].clamp_min(eps)
        newv = (A0[u] + dA_u).clamp_min(eps)
        S_corr = S_corr + float(ku) * (newv.pow(null_pow) - base.pow(null_pow))


    S_used = S0 + S_corr
    loss_null = (S_used.pow(2).sum()) / (4.0 * float(m)**2 * total_duration**(null_pow*2))
    """

    p_prev_det = p_prev.detach()
    temp_prev: Dict[int, torch.Tensor] = {}
    csc_terms = []

    def _csc_distance(cur, prev):
        if csc_norm == "l2":
            return torch.norm(cur - prev, p=2)
        elif csc_norm == "l1":
            return torch.norm(cur - prev, p=1)
        else:
            raise ValueError(f"Unsupported norm type: {csc_norm}")

    for i in range(B):
        u = int(src[i].item())
        cur_u = p_src[i]
        prev_u = temp_prev.get(u, p_prev_det[u])
        csc_terms.append(_csc_distance(cur_u, prev_u))
        temp_prev[u] = cur_u

        v = int(dst[i].item())
        cur_v = p_dst[i]
        prev_v = temp_prev.get(v, p_prev_det[v])
        csc_terms.append(_csc_distance(cur_v, prev_v))
        temp_prev[v] = cur_v

    loss_csc = torch.stack(csc_terms).mean() if len(csc_terms) > 0 else torch.zeros((), device=device, dtype=p_src.dtype)

    #loss = obs_weight * loss_obs + null_weight * loss_null + csc_weight * loss_csc 
    loss = -Q
    terms_raw = {
        "obs": loss_obs,
        "null": loss_null,
        "csc": loss_csc
    }
    loss_components = torch.stack([
        loss_obs.detach(),
        loss_null.detach(),
        loss_csc.detach()
    ])

    return loss, loss_components, terms_raw

