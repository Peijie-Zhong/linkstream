import torch
import math

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
    collapse_weight: float = 0.01,   # 建议从 0.01~1.0 试
    collapse_from: str = "pos",     # "pos" or "all"
    eps: float = 1e-12,
):
    """
    Base:
      pos = mean_i <p_src[i], p_dst[i]>
      w_ir = k_src[i] * k_neg[i,r]
      neg = weighted mean similarity to negatives
      loss_base = -(pos - lam*neg)

    Collapse regularization (DMoN-style):
      counts = sum over samples of p (per community mass)
      collapse = (sqrt(K)/M) * ||counts||_2 - 1, clamped >=0
      loss = loss_base + collapse_weight * collapse
    """
    B, K = p_src.shape
    assert p_dst.shape == (B, K)
    assert p_neg.dim() == 3 and p_neg.shape[0] == B and p_neg.shape[2] == K
    assert k_src.shape == (B,)
    assert k_neg.shape[:2] == p_neg.shape[:2]

    # ---- pos term ----
    pos_term = (p_src * p_dst).sum(dim=1).mean()

    # ---- weights ----
    w = (k_src.to(p_src.dtype).unsqueeze(1) * k_neg.to(p_src.dtype))  # [B,R]
    w = torch.clamp(w, min=0.0)
    if normalize_weights:
        w = w / (w.sum(dim=1, keepdim=True).clamp_min(eps))           # [B,R]

    # ---- neg term ----
    sim_src = (p_src.unsqueeze(1) * p_neg).sum(dim=2)                 # [B,R]
    neg_i_src = (w * sim_src).sum(dim=1)                              # [B]

    if symmetric:
        sim_dst = (p_dst.unsqueeze(1) * p_neg).sum(dim=2)             # [B,R]
        neg_i_dst = (w * sim_dst).sum(dim=1)                          # [B]
        neg_term = 0.5 * (neg_i_src.mean() + neg_i_dst.mean())
    else:
        neg_term = neg_i_src.mean()

    Q = pos_term - lam * neg_term
    loss_base = -Q

    # ---- collapse regularization ----
    # Choose which probabilities to regularize
    if collapse_from == "pos":
        P = torch.cat([p_src, p_dst], dim=0)      # [2B,K]
    elif collapse_from == "all":
        P = torch.cat([p_src, p_dst, p_neg.reshape(-1, K)], dim=0)  # [2B + BR, K]
    else:
        raise ValueError("collapse_from must be 'pos' or 'all'")

    M = P.shape[0]
    counts = P.sum(dim=0)                         # [K]
    collapse = (math.sqrt(K) / float(M)) * torch.norm(counts, p=2) - 1.0
    collapse = torch.clamp(collapse, min=0.0)

    loss = loss_base + collapse_weight * collapse

    debug = {
        "pos_term": float(pos_term.item()),
        "neg_term": float(neg_term.item()),
        "Q": float(Q.item()),
        "loss_base": float(loss_base.item()),
        "collapse": float(collapse.item()),
        "collapse_weight": float(collapse_weight),
        "collapse_from": collapse_from,
        "lam": float(lam),
        "normalize_weights": bool(normalize_weights),
        "avg_k_src": float(k_src.mean().item()),
        "avg_k_neg": float(k_neg.mean().item()),
        "avg_w_sum": float(w.sum(dim=1).mean().item()),
    }
    return loss, debug