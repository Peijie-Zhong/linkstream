import torch

def criterion_l_modularity(p_src, p_dst, 
                           T_src, T_neg, 
                           total_duration, 
                           last_p_src, 
                           omega=2.0):
    """
    修正后的 Loss (Strict EMM Version)
    
    参数:
        p_src: [B, K] 正样本当前概率 (需要优化)
        p_dst: [B, K] 目标节点当前概率
        # p_neg 被移除！EMM 不关心负样本现在在干嘛，只关心它过去在哪。
        
        T_src: [B, K] 源节点历史时长 (Context, Detached)
        T_neg: [B, S, K] 负样本历史时长 (Context, Detached)
    """
    
    # 1. 观测项 (Observed): 没变，最大化连边重合
    loss_obs = -torch.mean(torch.sum(p_src * p_dst, dim=-1))
    
    # 2. 期望项 (Null Model - EMM): 最小化期望增量
    # 梯度方向: p_src * sqrt(Tn / Tu)
    # T_src: [B, K] -> [B, 1, K]
    # T_neg: [B, S, K]
    
    # 加上 epsilon 防止除零
    epsilon = 1e-6
    
    # 计算排斥权重 (Repulsion Weight)
    # Weight [B, S, K]
    repulsion_weight = torch.sqrt(T_neg / (T_src.unsqueeze(1) + epsilon))
    
    # 归一化系数 (1 / 2|T|)
    coeff = 1.0 / (2.0 * total_duration)
    
    # 计算惩罚项
    # 我们希望 p_src 不要指向 repulsion_weight 大的社区
    # p_src: [B, 1, K] * weight: [B, S, K] -> [B, S, K]
    penalty_term = p_src.unsqueeze(1) * repulsion_weight
    
    # 求和 (对社区 K 和 负样本 S)
    # 也就是论文公式中的 sum_C (Expectation)
    loss_null = coeff * torch.mean(torch.sum(penalty_term, dim=[-1, -2]))
    
    # 3. 切换项 (Switch Cost)
    loss_csc = torch.mean(torch.sum((p_src - last_p_src) ** 2, dim=-1))
    
    total_loss = loss_obs + loss_null + omega * loss_csc
    
    return total_loss, loss_obs, loss_null, loss_csc


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # 2. 准备数据
    # -----------------------------------------------------------------------------
    time_links = [
        [2, 3, 0], [0, 1, 2], [2, 3, 3], [3, 4, 5], [2, 3, 6], [2, 4, 7],
        [0, 1, 8], [1, 2, 9], [3, 4, 9], [0, 2, 10], [1, 2, 11], [3, 4, 13],
        [1, 2, 14], [2, 4, 16], [0, 1, 17], [0, 1, 18], [2, 3, 18], [3, 4, 19],
    ]

    # 社区划分 (Node, Time) -> CommID
    comm_dict_raw = {
        0: {(3, 4), (4, 9), (3, 1), (3, 7), (4, 6), (4, 12), (3, 10), (3, 16), 
            (4, 15), (3, 13), (3, 19), (4, 18), (2, 2), (2, 5), (2, 17), (3, 0), 
            (4, 5), (3, 3), (3, 9), (4, 8), (4, 14), (3, 6), (3, 12), (4, 11), 
            (4, 17), (3, 18), (3, 15), (2, 4), (2, 1), (2, 7), (2, 16), (3, 2), 
            (4, 7), (3, 5), (3, 11), (4, 10), (4, 16), (3, 8), (3, 14), (4, 13), 
            (4, 19), (3, 17), (2, 0), (2, 3), (2, 6), (2, 18)}, 
        1: {(0, 2), (0, 5), (1, 6), (0, 8), (0, 14), (1, 3), (1, 9), (0, 11), 
            (0, 17), (2, 14), (1, 12), (1, 18), (2, 11), (1, 15), (0, 7), (1, 2), 
            (0, 4), (0, 10), (0, 16), (1, 5), (1, 11), (0, 13), (2, 10), (1, 8), 
            (1, 14), (2, 13), (1, 17), (0, 3), (0, 9), (1, 4), (0, 6), (0, 12), 
            (2, 9), (1, 7), (1, 13), (0, 15), (2, 12), (1, 10), (1, 16), (0, 18)}
    }

    # -----------------------------------------------------------------------------
    # 3. 数据预处理与 Tensor 构建
    # -----------------------------------------------------------------------------
    NUM_NODES = 5
    NUM_COMM = 2
    TOTAL_DURATION = 20.0 # 0到19

    # A. 构建查找表: (u, t) -> One-Hot Tensor
    lookup = {}
    # 初始化 T_context (统计每个节点在每个社区出现的次数作为时长)
    # Shape: [N, K]
    T_context_global = torch.zeros(NUM_NODES, NUM_COMM)

    for c_id, members in comm_dict_raw.items():
        for (u, t) in members:
            # 记录 One-Hot
            vec = torch.zeros(NUM_COMM)
            vec[c_id] = 1.0
            lookup[(u, t)] = vec
            
            # 累积时长 (简单计数，模拟 Sample-and-Hold 的结果)
            # 注意: 这里简单把出现一次当做时长+1
            T_context_global[u, c_id] += 1.0

    print("Global T (每个节点在各社区的累计计数):\n", T_context_global)

    # B. 构建 Batch Tensors
    # 我们把 time_links 当作一个 Batch 处理
    p_src_list = []
    p_dst_list = []
    T_src_list = []
    last_p_src_list = []

    # 辅助: 记录每个节点上一次的状态，用于计算 switch cost
    # 初始化为均匀分布
    node_last_state = {u: torch.ones(NUM_COMM)/NUM_COMM for u in range(NUM_NODES)}

    # 按时间遍历边
    for u, v, t in sorted(time_links, key=lambda x: x[2]):
        # 1. 获取当前概率 p_src, p_dst
        # 如果 (u,t) 在 comm 中，取 One-Hot；否则取均匀分布(或0)
        pu = lookup.get((u, t), torch.ones(NUM_COMM)/NUM_COMM)
        pv = lookup.get((v, t), torch.ones(NUM_COMM)/NUM_COMM)
        
        p_src_list.append(pu)
        p_dst_list.append(pv)
        
        # 2. 获取 T (Context)
        T_src_list.append(T_context_global[u])
        
        # 3. 获取 Last Prob (Switch Cost Target)
        last_p = node_last_state[u]
        last_p_src_list.append(last_p)
        
        # 更新状态
        node_last_state[u] = pu
        node_last_state[v] = pv # 假设 v 的状态也更新了

    # 转 Tensor
    p_src = torch.stack(p_src_list)
    p_dst = torch.stack(p_dst_list)
    T_src = torch.stack(T_src_list)
    last_p_src = torch.stack(last_p_src_list)

    # C. 构建负样本 (Negative Samples)
    # 为了测试，我们对每条边随机采 2 个负样本
    NUM_NEG = 10
    BATCH_SIZE = len(time_links)

    p_neg_list = []
    T_neg_list = []

    torch.manual_seed(42) # 固定随机性
    for i in range(BATCH_SIZE):
        u, _, t = time_links[i]
        
        # 随机采 2 个负样本
        negs = torch.randint(0, NUM_NODES, (NUM_NEG,))
        
        curr_p_neg = []
        curr_T_neg = []
        
        for n_idx in negs:
            n = n_idx.item()
            # 获取负样本在时刻 t 的概率
            pn = lookup.get((n, t), torch.ones(NUM_COMM)/NUM_COMM)
            curr_p_neg.append(pn)
            curr_T_neg.append(T_context_global[n])
        
        p_neg_list.append(torch.stack(curr_p_neg))
        T_neg_list.append(torch.stack(curr_T_neg))

    p_neg = torch.stack(p_neg_list) # [B, S, K]
    T_neg = torch.stack(T_neg_list) # [B, S, K]

    # -----------------------------------------------------------------------------
    # 4. 运行测试
    # -----------------------------------------------------------------------------
    print("\n--- Input Shapes ---")
    print(f"p_src: {p_src.shape}")
    print(f"p_neg: {p_neg.shape}")
    print(f"T_src: {T_src.shape}")

    loss, l_obs, l_null, l_csc = criterion_l_modularity(
        p_src, p_dst, 
        T_src, T_neg,
        TOTAL_DURATION,
        last_p_src,
        omega=2.0
    )

    print("\n--- Test Results ---")
    print(f"Observed Loss (连边重合度): {l_obs:.4f} (越接近 -1 越好)")
    print(f"Null Loss (期望惩罚):       {l_null:.4f} (越小越好)")
    print(f"Switch Cost (切换惩罚):     {l_csc:.4f}  (越小越好)")
    print(f"Total Loss:                 {loss:.4f}")

    # -----------------------------------------------------------------------------
    # 5. 结果分析 (Sanity Check)
    # -----------------------------------------------------------------------------
    # 验证 Observed Loss:
    # 检查你的 comm 数据，看连接的节点是否大多在同一社区
    same_comm_count = 0
    for u, v, t in time_links:
        # 简单查表，如果没有记录则视为不匹配
        c_u = -1
        c_v = -2
        for c, members in comm_dict_raw.items():
            if (u, t) in members: c_u = c
            if (v, t) in members: c_v = c
        
        if c_u == c_v and c_u != -1:
            same_comm_count += 1

    print(f"\n[验证] 真实数据中同社区边比例: {same_comm_count}/{len(time_links)} = {same_comm_count/len(time_links):.2f}")
    print(f"[验证] 模型计算的 Obs Loss: {l_obs:.4f}")
    if abs(l_obs - (-same_comm_count/len(time_links))) < 0.1:
        print(">> Obs Loss 计算合理！(考虑到部分节点无标签使用了均匀分布)")
    else:
        print(">> Obs Loss 偏差较大，请检查无标签节点的处理逻辑。")