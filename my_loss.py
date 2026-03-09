import torch
import math
import torch.nn.functional as F

def temporal_modularity_k_weighted_neg_loss(
    p_src: torch.Tensor,     # [B,K]
    p_dst: torch.Tensor,     # [B,K]
    p_neg: torch.Tensor,     # [B,R,K]
    k_src: torch.Tensor,     # [B]
    k_neg: torch.Tensor,     # [B,R]
    *,
    lam: float = 1.0,
    symmetric: bool = True,
    normalize_weights: bool = True,

    # --- collapse regularization ---
    collapse_weight: float = 0.01,
    collapse_from: str = "pos",

    # --- temporal smoothness (node-level) ---
    smooth_weight: float = 0.0,
    smooth_mode: str = "l2",
    p_prev: torch.Tensor | None = None,
    nodes_src: torch.Tensor | None = None,
    nodes_dst: torch.Tensor | None = None,
    dt_src: torch.Tensor | None = None,
    dt_dst: torch.Tensor | None = None,
    smooth_time_beta: float | None = None,

    eps: float = 1e-12,
):

    B, K = p_src.shape
    assert p_dst.shape == (B, K)
    assert p_neg.dim() == 3 and p_neg.shape[0] == B and p_neg.shape[2] == K
    assert k_src.shape == (B,)
    assert k_neg.shape[:2] == p_neg.shape[:2]


    pos_term = (p_src * p_dst).sum(dim=1).mean()


    w = (k_src.to(p_src.dtype).unsqueeze(1) * k_neg.to(p_src.dtype))
    w = torch.clamp(w, min=0.0)
    if normalize_weights:
        w = w / (w.sum(dim=1, keepdim=True).clamp_min(eps))  


    sim_src = (p_src.unsqueeze(1) * p_neg).sum(dim=2)
    neg_i_src = (w * sim_src).sum(dim=1)

    if symmetric:
        sim_dst = (p_dst.unsqueeze(1) * p_neg).sum(dim=2) 
        neg_i_dst = (w * sim_dst).sum(dim=1)
        neg_term = 0.5 * (neg_i_src.mean() + neg_i_dst.mean())
    else:
        neg_term = neg_i_src.mean()

    Q = pos_term - lam * neg_term
    loss_base = -Q

    # ---- collapse regularization ----
    if collapse_from == "pos":
        P = torch.cat([p_src, p_dst], dim=0)
    elif collapse_from == "all":
        P = torch.cat([p_src, p_dst, p_neg.reshape(-1, K)], dim=0)
    else:
        raise ValueError("collapse_from must be 'pos' or 'all'")

    M = P.shape[0]
    counts = P.sum(dim=0)
    collapse = (math.sqrt(K) / float(M)) * torch.norm(counts, p=2) - 1.0
    collapse = torch.clamp(collapse, min=0.0)

    # ---- temporal smoothness ----
    smooth = torch.zeros((), device=p_src.device, dtype=p_src.dtype)
    if smooth_weight > 0.0:
        if p_prev is None or nodes_src is None or nodes_dst is None:
            raise ValueError("smooth_weight>0 requires p_prev, nodes_src, nodes_dst")

        # concat occurrences
        nodes_all = torch.cat([nodes_src, nodes_dst], dim=0)          # [2B]
        P_all = torch.cat([p_src, p_dst], dim=0)                      # [2B,K]
        prev_all = p_prev[nodes_all].to(dtype=P_all.dtype)            # [2B,K]

        if smooth_time_beta is not None:
            if dt_src is None or dt_dst is None:
                raise ValueError("smooth_time_beta given, but dt_src/dt_dst is None")
            dt_all = torch.cat([dt_src, dt_dst], dim=0).to(dtype=P_all.dtype)  # [2B]
            wt = torch.exp(-float(smooth_time_beta) * torch.clamp(dt_all, min=0.0))  # [2B]
        else:
            wt = None

        if smooth_mode == "l2":
            d = ((P_all - prev_all) ** 2).sum(dim=1)                  # [2B]
        elif smooth_mode == "kl":
            # KL(P_all || prev_all) （prev 是“上一时刻”分布，不需要 grad）
            p = torch.clamp(P_all, min=eps)
            q = torch.clamp(prev_all, min=eps)
            d = (p * (p.log() - q.log())).sum(dim=1)                  # [2B]
        else:
            raise ValueError("smooth_mode must be 'l2' or 'kl'")

        if wt is not None:
            smooth = (wt * d).mean()
        else:
            smooth = d.mean()

    loss = loss_base + collapse_weight * collapse + smooth_weight * smooth

    debug = {
        "pos_term": float(pos_term.item()),
        "neg_term": float(neg_term.item()),
        "Q": float(Q.item()),
        "loss_base": float(loss_base.item()),
        "collapse": float(collapse.item()),
        "collapse_weight": float(collapse_weight),
        "smooth": float(smooth.item()),
        "smooth_weight": float(smooth_weight),
        "smooth_mode": smooth_mode,
        "lam": float(lam),
        "normalize_weights": bool(normalize_weights),
        "avg_k_src": float(k_src.mean().item()),
        "avg_k_neg": float(k_neg.mean().item()),
        "avg_w_sum": float(w.sum(dim=1).mean().item()),
    }
    return loss, debug