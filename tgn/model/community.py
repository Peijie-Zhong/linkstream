import torch.nn as nn
import torch.nn.functional as F
import torch

class CommunityProjector(nn.Module):
    def __init__(self, embedding_dim, num_communities, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_communities),
            nn.Softmax(dim=-1) # 输出概率分布
        )
    
    def forward(self, x):
        return self.mlp(x)
    

def compute_l_modularity_loss(p_src, p_dst, p_neg, 
                              T_src, T_neg, 
                              total_duration, 
                              last_p_src, 
                              omega=1.0):
    """
    p_src, p_dst: [B, K] 正样本概率
    p_neg: [B, S, K] 负样本概率
    T_src: [B, K] 源节点历史时长
    T_neg: [B, S, K] 负样本历史时长
    """
    # 1. 观测项 (Observed): 最大化正样本重合度 -> 最小化负点积
    # Q_obs = sum(p_u * p_v)
    loss_obs = -torch.mean(torch.sum(p_src * p_dst, dim=-1))
    
    # 2. 期望项 (Expected - EMM): 最小化负样本重合度 (考虑时间权重)
    # 权重 W = sqrt(Tu * Tn) / |T|
    # 注意维度: T_src [B, 1, K], T_neg [B, S, K]
    time_weight = torch.sqrt(T_src.unsqueeze(1) * T_neg) / total_duration
    
    # Dot product: p_src [B, 1, K] * p_neg [B, S, K] -> Sum over K -> [B, S]
    dot_null = torch.sum(p_src.unsqueeze(1) * p_neg, dim=-1)
    
    # 加上权重
    loss_null = torch.mean(time_weight * dot_null)
    
    # 3. 平滑项 (Switch Cost)
    # 简单 MSE 约束
    loss_csc = torch.mean((p_src - last_p_src) ** 2)
    
    return loss_obs + loss_null + omega * loss_csc