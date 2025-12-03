import logging
import numpy as np
import torch
from collections import defaultdict

from tgn.utils.utils import MergeLayer
from tgn.modules.memory import Memory
from tgn.modules.message_aggregator import get_message_aggregator
from tgn.modules.message_function import get_message_function
from tgn.modules.memory_updater import get_memory_updater
from tgn.modules.embedding_module import get_embedding_module
from tgn.model.time_encoding import TimeEncode

import torch.nn as nn
import torch.nn.functional as F


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
    

class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift=0, std_time_shift=1,
               n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False,
               # additional parameters can be added here
               num_communities=5
               ):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep
    
    # my changes
    self.num_communities = num_communities
    # end my changes

    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.memory = None

    self.mean_time_shift = mean_time_shift
    self.std_time_shift = std_time_shift

    if self.use_memory:
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors)
    
    self.community_projector = CommunityProjector(embedding_dim=self.embedding_dimension,
                                                  num_communities=self.num_communities,
                                                  dropout=dropout)

  def compute_temporal_embeddings(self, nodes, timestamps, n_neighbors=20):
      """
      通用 Embedding 计算函数。
      不再区分 source/destination/negative，而是对传入的一批节点和时间计算 Embedding。
      
      Args:
          nodes: [batch_size] 或 [any_size] 的节点 ID (numpy array 或 tensor)
          timestamps: [batch_size] 对应的交互时间戳 (numpy array 或 tensor)
          n_neighbors: 邻居采样数
      Returns:
          node_embedding: [batch_size, embedding_dim]
      """
      
      # -------------------------------------------------------
      # 1. 输入数据标准化 (Data Standardization)
      # -------------------------------------------------------
      # TGN 内部通常需要 Tensor，且位于正确的 device 上
      if isinstance(nodes, np.ndarray):
          nodes_tensor = torch.from_numpy(nodes).long().to(self.device)
      else:
          nodes_tensor = nodes.long().to(self.device)

      if isinstance(timestamps, np.ndarray):
          timestamps_tensor = torch.from_numpy(timestamps).float().to(self.device)
      else:
          timestamps_tensor = timestamps.float().to(self.device)
          
      # 为了配合 embedding_module 的接口，这里还需要保留 numpy 版本的 nodes
      # 因为 neighbor_finder 通常操作 numpy 数组
      nodes_numpy = nodes.cpu().numpy() if isinstance(nodes, torch.Tensor) else nodes
      timestamps_numpy = timestamps.cpu().numpy() if isinstance(timestamps, torch.Tensor) else timestamps

      # -------------------------------------------------------
      # 2. 记忆读取与时间差计算 (Memory Retrieval & Time Diffs)
      # -------------------------------------------------------
      memory = None
      time_diffs = None

      if self.use_memory:
        # A. 处理 Memory 更新 (仅在 Batch 开始时触发一次)
        # 如果之前的 Batch 留下了 Message，这里会先把它们应用到 Memory 上
        if self.memory_update_at_start:
          # 注意：这里传入的是全量节点 range(n_nodes)，因为 update 可能涉及任何节点
          memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                        self.memory.messages)
        else:
          # 如果不更新，直接读取当前状态
          memory = self.memory.get_memory(list(range(self.n_nodes)))
          last_update = self.memory.last_update

        # B. 计算 Time Diffs (当前交互时间 - 上次记忆更新时间)
        # 这是 TGN 处理时间间隔特征的关键
        
        # 获取传入节点的上次更新时间
        last_update_of_nodes = last_update[nodes_tensor].float().to(self.device)
        
        # 计算差值
        time_diffs = timestamps_tensor - last_update_of_nodes
        
        # 归一化 (Normalization)
        # 使用源节点的统计量 (mean_time_shift_src) 作为通用标准
        # 在原代码中 src/dst/neg 分别处理，这里统一视为"参与交互的节点"
        time_diffs = (time_diffs - self.mean_time_shift) / self.std_time_shift

      # -------------------------------------------------------
      # 3. 调用核心嵌入模块 (Embedding Module)
      # -------------------------------------------------------
      # 这一步执行图注意力 (GAT) 或 图卷积
      # 注意：self.embedding_module 内部需要 numpy 格式的 nodes 来做邻居采样
      node_embedding = self.embedding_module.compute_embedding(
          memory=memory,
          source_nodes=nodes_numpy,     # 传入 numpy 用于采样
          timestamps=timestamps_numpy,  # 传入 numpy 用于采样
          n_layers=self.n_layers,
          n_neighbors=n_neighbors,
          time_diffs=time_diffs         # 传入 Tensor 用于特征拼接
      )
      return node_embedding


  def compute_community_prob(self, nodes, timestamps, n_neighbors=20):
      embeddings = self.compute_temporal_embeddings(nodes, timestamps, n_neighbors)
      return self.community_projector(embeddings)
  

  def update_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)
    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)
    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)
    

  def get_updated_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
