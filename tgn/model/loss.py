import torch
from typing import Dict
import math


def longitudinal_modularity_loss(
    p_src, p_dst, src, dst, node_lifespan, global_degree, m, total_duration,
    p_prev, delta_src, delta_dst,
    csc_norm="l2",
    null_weight=1.0,
    csc_weight=1.0,
    collapse_weight=1.0
):
    device = p_src.device
    B = p_src.size(0)
    K = int(p_src.size(1))

    # -------------------------
    # 1) soft longitudinal modularity (batch as a small link stream)
    #    Observed term: sum_i <p_src[i], p_dst[i]>
    #    Expected term (user-specified):
    #      sum_c sum_{u,v} (k_u k_v)/(2m) * (sqrt(T_u^c T_v^c))/T
    #    where T_u^c is accumulated from time-varying p using delta_src/dst.
    # -------------------------

    # Observed (events) term
    obs_term = (p_src * p_dst).sum(dim=1).mean()

    # Build per-node time-in-community T_u^c within this batch using deltas
    # occurrences: src at t contributes p_src * delta_src; dst contributes p_dst * delta_dst
    nodes_all = torch.cat([src, dst], dim=0).to(device=device)
    P_all = torch.cat([p_src, p_dst], dim=0)  # [2B, K]
    dt_all = torch.cat([delta_src, delta_dst], dim=0).to(dtype=p_src.dtype)  # [2B]

    # Guard against non-positive deltas (should not happen; keep stable)
    dt_all = torch.clamp(dt_all, min=0.0)

    # T_contrib[o, c] = p(o,c) * dt(o)
    T_contrib = P_all * dt_all[:, None]  # [2B, K]

    # unique nodes in this batch
    uniq_nodes, inv = torch.unique(nodes_all, return_inverse=True)
    U = int(uniq_nodes.numel())

    # Accumulate T_u^c  (U x K)
    T_u = torch.zeros((U, K), device=device, dtype=p_src.dtype)
    T_u.index_add_(0, inv, T_contrib)

    # Also accumulate total observed duration per node (for collapse regularization)
    dur_u = torch.zeros((U,), device=device, dtype=p_src.dtype)
    dur_u.index_add_(0, inv, dt_all)

    # Convert global_degree (python list/np array) -> tensor for uniq nodes
    # NOTE: global_degree is assumed to be indexable by raw node id.
    k_u = p_src.new_tensor([float(global_degree[int(u.item())]) for u in uniq_nodes])

    # Expected term in closed form:
    #   (1 / ((2m) * T)) * sum_c (sum_u k_u * sqrt(T_u^c))^2
    # Here we use total_duration from args as T.
    T_denom = float(total_duration) if float(total_duration) > 0 else 1.0

    sqrt_T_u = torch.sqrt(T_u)
    #S_c = (k_u[:, None] * sqrt_T_u).sum(dim=0)
    S_c = (k_u[:, None] * T_u).sum(dim=0)
    print("Sc:", S_c)
    exp_term = (S_c * S_c).sum() / (2.0 * float(m) * T_denom**2)

    exp_term = exp_term / (2*B)
    print("obs:", obs_term)
    print("exp_term:", exp_term)
    Q = obs_term - exp_term
    loss_modularity = -Q

    # For collapse regularization downstream, define a node-level soft assignment for this batch:
    # time-weighted average p_u = (sum_occ p*dt) / (sum_occ dt)
    P_node = T_u / (dur_u[:, None] + 1e-12)  # [U, K]
    P_occ = P_node
    M_occ = U

    # -------------------------
    # 2) CSC temporal smoothness
    # -------------------------
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

    loss_csc = (
        torch.stack(csc_terms).mean()
        if len(csc_terms) > 0
        else torch.zeros((), device=device, dtype=p_src.dtype)
    )
    """

    # -----------------------------------------
    # 3) collapse regularization
    # -----------------------------------------
    # DMoN 里是: (sqrt(K)/n) * || C^T 1 ||_F - 1
    # 这里用 batch-node-level 的 soft assignment P_occ (U x K)
    # counts[k] = sum_u P_occ[u,k]
    # 坍缩到单一簇 => counts 的 L2 范数最大（=U）；均匀分配 => 更小
    if P_occ is None:
        loss_collapse = torch.zeros((), device=device, dtype=p_src.dtype)
    else:
        counts = P_occ.sum(dim=0)  # [K]
        loss_collapse = (math.sqrt(K) / float(M_occ)) * torch.norm(counts, p=2) - 1.0
        loss_collapse = torch.clamp(loss_collapse, min=0.0)

    # -------------------------
    # total loss
    # -------------------------
    loss = (
        null_weight * loss_modularity
        # + csc_weight * loss_csc
        + collapse_weight * loss_collapse
    )

    terms_raw = {
        "modularity": loss_modularity,
        # "csc": loss_csc,
        "collapse": loss_collapse,
    }
    loss_components = torch.stack([
        loss_modularity.detach(),
        # loss_csc.detach(),
        loss_collapse.detach(),
    ])

    return loss, loss_components, terms_raw


"""

import torch
from typing import Dict
import math


def longitudinal_modularity_loss(
    p_src, p_dst, src, dst, node_lifespan, global_degree, m, total_duration,
    p_prev, delta_src, delta_dst,
    csc_norm="l2",
    null_weight=1.0,
    csc_weight=1.0,
    collapse_weight=1.0
):
    device = p_src.device
    B = p_src.size(0)
    K = int(p_src.size(1))

    P_list = []      # list of p_o in R^K
    S_list = []      # list of s_o scalars (k_u * alpha_u)
    edge_occ_pairs = []

    def _append_occ(p_u: torch.Tensor, u: int, dt: torch.Tensor) -> int:
        dtv = dt.to(dtype=p_u.dtype)
        if dtv.item() <= 0:
            raise ValueError("Negative delta time.")

        ku = float(global_degree[u])
        Tu = float(node_lifespan[u])

        alpha = float(math.sqrt(dtv.item() / (Tu+1)))
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

    # -------------------------
    # 1) modularity-style term
    # -------------------------
    if M_occ <= 1:
        loss_modularity = torch.zeros((), device=device, dtype=p_src.dtype)
        P_occ = None
    else:
        P_occ = torch.stack(P_list, dim=0)  # [M_occ, K]
        S_occ = torch.stack(S_list, dim=0)  # [M_occ]

        Sim = P_occ @ P_occ.t()             # [M_occ, M_occ]

        A_occ = torch.zeros((M_occ, M_occ), device=device, dtype=p_src.dtype)
        for (a, b) in edge_occ_pairs:
            A_occ[a, b] = A_occ[a, b] + 1.0
            A_occ[b, a] = A_occ[b, a] + 1.0
        #E_occ = (S_occ[:, None] * S_occ[None, :]) / S_occ.sum()
        E_occ = (S_occ[:, None] * S_occ[None, :]) / (2 * m)
        # remove diagonal
        A_occ = A_occ - torch.diag(torch.diag(A_occ))
        E_occ = E_occ - torch.diag(torch.diag(E_occ))
        Sim = Sim - torch.diag(torch.diag(Sim))

        Q = ((A_occ - E_occ) * Sim).sum() / (2 * B)
        print(((A_occ) * Sim).sum() / (2 * B))
        print((E_occ * Sim).sum() / (2 * B))


        loss_modularity = -Q

    # -------------------------
    # 2) CSC temporal smoothness
    # -------------------------

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

    loss_csc = (
        torch.stack(csc_terms).mean()
        if len(csc_terms) > 0
        else torch.zeros((), device=device, dtype=p_src.dtype)
    )

    # -----------------------------------------
    # 3) collapse regularization
    # -----------------------------------------
    # DMoN 里是: (sqrt(K)/n) * || C^T 1 ||_F - 1
    # 这里用 occurrence-level 的 soft assignment P_occ (M_occ x K)
    # counts[k] = sum_o P_occ[o,k]
    # 坍缩到单一簇 => counts 的 L2 范数最大（=M_occ）；均匀分配 => 更小
    if P_occ is None:
        loss_collapse = torch.zeros((), device=device, dtype=p_src.dtype)
    else:
        counts = P_occ.sum(dim=0)  # [K]
        loss_collapse = (math.sqrt(K) / float(M_occ)) * torch.norm(counts, p=2) - 1.0

        loss_collapse = torch.clamp(loss_collapse, min=0.0)

    # -------------------------
    # total loss
    # -------------------------
    loss = (
        null_weight * loss_modularity
        #+ csc_weight * loss_csc
        + collapse_weight * loss_collapse
    )

    terms_raw = {
        "modularity": loss_modularity,
        #"csc": loss_csc,
        "collapse": loss_collapse, 
    }
    loss_components = torch.stack([
        loss_modularity.detach(),
        #loss_csc.detach(),
        loss_collapse.detach(),
    ])

    return loss, loss_components, terms_raw

"""