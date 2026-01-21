import torch
from typing import Dict, Tuple


def longitudinal_modularity_loss(p_src, p_dst, 
                                 src, dst,
                                 delta_src, delta_dst,
                                 A_base, S_base, 
                                 node_lifespans, global_degree,
                                 m, total_duration,
                                 p_prev,
                                 omega=2.0):
    B, K = p_src.shape
    loss_obs = (p_src * p_dst).sum(dim=1).mean()

    delta_a_nodes: Dict[int, torch.Tensor] = {}
    last_p_nodes: Dict[int, torch.Tensor] = {}

    def _accumulate(node_ids: torch.Tensor, probs: torch.Tensor, deltas: torch.Tensor):
        for i in range(node_ids.numel):
            u = int(node_ids[i].item())
            dt = deltas[i]
            if dt.numel() == 0:
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

    S_used = S0.clone()

    for u, delta_a_u in delta_a_nodes.items():
        ku = global_degree[u]
        Tu = node_lifespans[u].clamp_min(1.0)
        corr = ku * (torch.sqrt(A0[u] +delta_a_u) - torch.sqrt(A0[u]))
        S_used[u] = S0[u] + corr

    denom = 2.0 * float(m) * total_duration ** 0.5
    loss_null = S_used.pow(2).sum() / denom


    p_prev_det = p_prev.detach()
    if len(last_p_nodes) == 0:
        loss_csc = torch.tensor(0.0, device=p_src.device)
    else:
        csc_vals = []
        for u, last_p in last_p_nodes.items():
            prev = p_prev_det

            c = torch.norm(last_p - prev, p=2)
            csc_vals.append(c)

        loss_csc = torch.stack(csc_vals).mean()    
    loss = loss_obs + loss_null + omega * loss_csc
    loss_components = torch.stack([loss_obs.detach(), loss_null.detach(), loss_csc.detach()])

    return loss, last_p_nodes, loss_components

