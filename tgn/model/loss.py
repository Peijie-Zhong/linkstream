import torch
from typing import Dict
import math


def longitudinal_modularity_loss(
    p_src, p_dst, src, dst, node_lifespan, global_degree, m, total_duration,
    p_prev, delta_src, delta_dst,
    csc_norm="l2",
    null_weight=1.0,
    csc_weight=1.0,
    collapse_weight=1.0,
    neg_weight=0.3,        # <<< 新增：负采样损失权重
    neg_samples=10,        # <<< 新增：每条边采多少个负样本
    neg_max_tries=5,       # <<< 新增：拒绝采样最大重采次数
):
    device = p_src.device
    B = p_src.size(0)
    K = int(p_src.size(1))

    # -------- time factor per endpoint --------
    Tu_src = node_lifespan[src].to(device=device, dtype=p_src.dtype)
    Tu_dst = node_lifespan[dst].to(device=device, dtype=p_src.dtype)
    alpha_src = (delta_src.to(p_src.dtype) / (Tu_src + 1.0))
    alpha_dst = (delta_dst.to(p_dst.dtype) / (Tu_dst + 1.0))

    # -------- batch degree (within-batch) --------
    all_nodes = torch.cat([src, dst], dim=0)  # [2B]
    uniq_nodes, inv_all = torch.unique(all_nodes, return_inverse=True)
    inv_src = inv_all[:B]                      # [B]
    inv_dst = inv_all[B:]                      # [B]

    deg_uniq = torch.bincount(inv_all, minlength=uniq_nodes.numel()).to(device=device, dtype=p_src.dtype)
    k_src = deg_uniq[inv_src]                  # [B]
    k_dst = deg_uniq[inv_dst]                  # [B]
    two_m_b = deg_uniq.sum()                   # == 2*m_b == 2B (无向、每交互一条边)

    # -------- modularity pieces (你当前版本的 A - E) --------
    # A: observed similarity on edges
    sim_e = (p_src * p_dst).sum(dim=1)         # [B]
    A = sim_e.mean()

    # E: stub-level implicit all-pairs (time factor enters via sqrt(alpha))
    p_stub = torch.cat([p_src, p_dst], dim=0)                  # [2B, K]
    alpha_stub = torch.cat([alpha_src, alpha_dst], dim=0)      # [2B]
    w_stub = torch.sqrt(torch.clamp(alpha_stub, min=0.0) + 1e-12).unsqueeze(1)  # [2B,1]
    s_c = (w_stub * p_stub).sum(dim=0)                         # [K]

    # 这里给一个更稳定的尺度：E1 = (Σ s_c^2) / (2m)  是 O(B)；再除一次 (2m) 让它变 O(1) 跟 A 对齐
    E = (s_c.pow(2).sum()) / (two_m_b + 1e-12)
    E = E / (two_m_b + 1e-12)

    Q = A - E
    print("E:", E)
    print("A:", A)
    loss_modularity = -Q

    # -------- collapse regularization (same as yours) --------
    P_occ = torch.cat([p_src, p_dst], dim=0)   # [2B, K]
    M_occ = P_occ.size(0)
    counts = P_occ.sum(dim=0)
    loss_collapse = (math.sqrt(K) / float(M_occ)) * torch.norm(counts, p=2) - 1.0
    loss_collapse = torch.clamp(loss_collapse, min=0.0)
    print("collapse:", loss_collapse)

    # =========================
    #   Negative sampling loss
    # =========================
    # 目标：对 batch 内“无边对”施加惩罚，推开它们的社区相似度 sim(u,v)
    loss_neg = torch.zeros((), device=device, dtype=p_src.dtype)

    if neg_weight > 0.0 and neg_samples > 0 and uniq_nodes.numel() > 1:
        U = uniq_nodes.numel()

        # (1) 先构造 node-level 平均概率 p_node[uidx]，用于负样本节点的 p
        p_sum = torch.zeros(U, K, device=device, dtype=p_src.dtype)
        cnt = torch.zeros(U, device=device, dtype=p_src.dtype)
        p_sum.index_add_(0, inv_all, p_stub)  # inv_all: [2B], p_stub: [2B,K]
        cnt.index_add_(0, inv_all, torch.ones_like(inv_all, dtype=p_src.dtype))
        p_node = p_sum / (cnt.unsqueeze(1) + 1e-12)  # [U,K]

        # (2) hash batch 内真实边，避免采到真实边（无向：双向都存）
        edge_hash = torch.cat([
            inv_src * U + inv_dst,
            inv_dst * U + inv_src,
        ], dim=0)
        edge_hash = torch.unique(edge_hash)

        # (3) 为每条边采 neg_samples 个候选负节点（index in [0,U)）
        cand = torch.randint(0, U, (B, neg_samples), device=device)

        # (4) 拒绝采样：避开 self / 真邻居 / batch 内真实边
        for _ in range(int(neg_max_tries)):
            bad_self_src = cand.eq(inv_src.unsqueeze(1)) | cand.eq(inv_dst.unsqueeze(1))
            bad_self_dst = cand.eq(inv_dst.unsqueeze(1)) | cand.eq(inv_src.unsqueeze(1))

            h_src = inv_src.unsqueeze(1) * U + cand
            h_dst = inv_dst.unsqueeze(1) * U + cand

            bad_edge_src = torch.isin(h_src, edge_hash)
            bad_edge_dst = torch.isin(h_dst, edge_hash)

            bad = bad_self_src | bad_self_dst | bad_edge_src | bad_edge_dst
            if not bad.any():
                break
            cand = torch.where(
                bad,
                torch.randint(0, U, cand.shape, device=device),
                cand
            )

        # (5) 计算负相似度：用端点的 occurrence-level p 与负节点的 node-level 平均 p
        p_cand = p_node[cand]  # [B, neg_samples, K]

        sim_neg_src = (p_src.unsqueeze(1) * p_cand).sum(dim=-1)  # [B, neg_samples]
        sim_neg_dst = (p_dst.unsqueeze(1) * p_cand).sum(dim=-1)  # [B, neg_samples]

        loss_neg = 0.5 * (sim_neg_src.mean() + sim_neg_dst.mean())
        print("neg:", loss_neg)

    # -------- total loss --------
    loss = (
        null_weight * loss_modularity
        + collapse_weight * loss_collapse
        + neg_weight * loss_neg
    )
    return loss
    """
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

        alpha = float(dtv.item() / (Tu+1))
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
        E_occ = (S_occ[:, None] * S_occ[None, :]) / S_occ.sum()

        # remove diagonal
        A_occ = A_occ - torch.diag(torch.diag(A_occ))
        E_occ = E_occ - torch.diag(torch.diag(E_occ))
        Sim = Sim - torch.diag(torch.diag(Sim))

        Q = ((A_occ - E_occ) * Sim).sum() / (2*B)
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
