import torch
from typing import Dict, Tuple
import logging

def longitudinal_modularity_loss(p_src, p_dst, src, dst, delta_a_nodes,
                                 A_base, S_base, global_degree,m, total_duration, 
                                 p_prev, csc_norm="l2", obs_weight=1.0,
                                 null_weight=1.0, csc_weight=1.0, collapse_weight=0.1):
    device = p_src.device
    B = p_src.size(0)
    # observation loss
    loss_obs = -(p_src * p_dst).sum(dim=1).mean()
    # null model loss
    eps = 1e-12
    A0 = A_base.detach()   # [N,K] baseline A_old
    S0 = S_base.detach()   # [K]   baseline S_old

    S_corr = torch.zeros_like(S0)  # [K]


    for u, dA_u in delta_a_nodes.items():
        ku = global_degree[u]
        base = A0[u].clamp_min(eps)               # [K]
        newv = (A0[u] + dA_u).clamp_min(eps)      # [K]
        S_corr = S_corr + ku * (torch.sqrt(newv) - torch.sqrt(base))

    S_used = S0 + S_corr
    loss_null = (S_used.pow(2).sum()) / (4.0 * float(m)**2 * total_duration)


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

    loss = obs_weight * loss_obs + null_weight * loss_null + csc_weight * loss_csc 

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


    loss_components = torch.stack([loss_obs.detach().sum(), 
                                   loss_null.detach(), 
                                   loss_csc.detach()])

    return loss, loss_components, terms_raw
#, extra
