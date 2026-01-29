import torch
from typing import Dict, Tuple


def longitudinal_modularity_loss(p_src, p_dst, src, dst, delta_src, delta_dst,
                                 A_base, S_base, global_degree,m, total_duration, 
                                 p_prev, csc_norm = "l2", obs_weight=1.0,
                                 null_weight=1.0, csc_weight=1.0, collapse_weight=0.1,
                                 conf_weight=0.1):
    device = p_src.device
    B = p_src.size(0)

    loss_obs = -(p_src * p_dst).sum(dim=1).mean()

    delta_a_nodes: Dict[int, torch.Tensor] = {}
    last_p_nodes: Dict[int, torch.Tensor] = {}

    def _accumulate(node_ids: torch.Tensor, probs: torch.Tensor, deltas: torch.Tensor):
        for i in range(node_ids.numel()):
            u = int(node_ids[i].item())
            dt = deltas[i]
            if dt.item() <= 0:
                print("Warning: non-positive time delta encountered:", dt.item())
                continue
            dt_val = dt.to(dtype=probs.dtype)
            inc = probs[i] * dt_val
            if u in delta_a_nodes:
                delta_a_nodes[u] = delta_a_nodes[u] + inc
            else:
                delta_a_nodes[u] = inc
            last_p_nodes[u] = probs[i]

    _accumulate(src, p_src, delta_src)
    _accumulate(dst, p_dst, delta_dst)

    A0 = A_base.detach()
    S0 = S_base.detach()
    dtype = S0.dtype
    DeltaS = torch.zeros_like(S0)  # [K]
    for u, delta_a_u in delta_a_nodes.items():
        ku = global_degree[u].to(dtype=dtype)  # scalar (ensure dtype)
        # A0[u] : [K], delta_a_u : [K]
        # ΔS_u = k_u( sqrt(A0+ΔA) - sqrt(A0) )
        DeltaS = DeltaS + ku * (torch.sqrt(A0[u] + delta_a_u) - torch.sqrt(A0[u]))

    # ΔF = ||S0+ΔS||^2 - ||S0||^2 = 2<S0,ΔS> + ||ΔS||^2
    delta_energy = 2.0 * (S0 * DeltaS).sum() + (DeltaS * DeltaS).sum()

    denom = max(2.0 * float(m) * total_duration, 1.0)
    loss_null = delta_energy / denom
    
    extra = {
        "DeltaS": DeltaS.detach(),          # 仅用于打印
        "delta_energy": delta_energy.detach()
    }
    

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

    p_all = torch.cat([p_src, p_dst], dim=0)  # [2B, K]
    q = p_all.mean(dim=0)                     # [K]
    K = q.numel()
    eps = 1e-8
    uniform = torch.full_like(q, 1.0 / K)

    loss_balance = torch.sum(q * torch.log((q + eps) / uniform))

    entropy = -(p_all * (p_all + eps).log()).sum(dim=1).mean()  # scalar
    loss_conf = entropy  

    loss = obs_weight * loss_obs + null_weight * loss_null + csc_weight * loss_csc + \
        collapse_weight * loss_balance + conf_weight * loss_conf
    
    
    # 现有：loss_obs, loss_null, loss_csc, loss_balance, loss_conf 都是带图的
    terms_raw = {
        "obs": loss_obs,
        "null": loss_null,
        "csc": loss_csc,
        "balance": loss_balance,
        "conf": loss_conf,
    }

    # 你原来的日志用的 detach 组件继续保留
    loss_components = torch.stack([
        loss_obs.detach(),
        loss_null.detach(),
        loss_csc.detach(),
        loss_balance.detach(),
        loss_conf.detach(),
    ])


    loss_components = torch.stack([loss_obs.detach().sum(), 
                                   loss_null.detach(), 
                                   loss_csc.detach(), 
                                   loss_balance.detach(), 
                                   loss_conf.detach()])

    return loss, last_p_nodes, loss_components, delta_a_nodes, terms_raw, extra
