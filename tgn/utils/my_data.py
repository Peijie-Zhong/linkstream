import numpy as np
import random
import pandas as pd
import os

class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)


def get_data(dataset_name, randomize_features=True):
  DEFAULT_DIM = 16
  ### Load data
  graph_df = pd.read_csv('./data/{}.csv'.format(dataset_name))

  sources = graph_df.source.values
  destinations = graph_df.destination.values
  edge_idxs = graph_df.idx.values
  timestamps = graph_df.timestamp.values

  max_node_idx = max(sources.max(), destinations.max())
  num_nodes = max_node_idx + 1
  num_edges = len(graph_df)

  edge_feat_path = './data/{}.npy'.format(dataset_name)
  if os.path.exists(edge_feat_path):
      print(f"Loading edge features: {edge_feat_path}")
      edge_features = np.load(edge_feat_path)
  else:
      print(f"cannot find ({edge_feat_path}), use zero-vector (dim={DEFAULT_DIM})...")
      edge_features = np.zeros((num_edges, DEFAULT_DIM), dtype=np.float32)

  node_feat_path = './data/{}_node.npy'.format(dataset_name)
  if os.path.exists(node_feat_path):
      print(f"cannot find node feature: {node_feat_path}")
      node_features = np.load(node_feat_path)
  else:
      print(f"cannot find node feature: {node_feat_path}), use zero vector(dim={DEFAULT_DIM})...")
      node_features = np.zeros((num_nodes, DEFAULT_DIM), dtype=np.float32)

  full_data = Data(sources, destinations, timestamps, edge_idxs)

  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))

  
  return node_features, edge_features, full_data

def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp = dict()
  all_timediffs = []

  for k in range(len(sources)):
      source_id = sources[k]
      dest_id = destinations[k]
      c_timestamp = timestamps[k]

      # --- 处理 Source 节点 ---
      if source_id not in last_timestamp:
          last_timestamp[source_id] = 0
      
      # 计算该节点距离上次交互过了多久
      all_timediffs.append(c_timestamp - last_timestamp[source_id])
      
      # 更新该节点的最新时间
      last_timestamp[source_id] = c_timestamp

      # --- 处理 Destination 节点 ---
      if dest_id not in last_timestamp:
          last_timestamp[dest_id] = 0
          
      all_timediffs.append(c_timestamp - last_timestamp[dest_id])
      last_timestamp[dest_id] = c_timestamp

  # 2. 计算全局统计量
  mean_time_shift = np.mean(all_timediffs)
  std_time_shift = np.std(all_timediffs)

  # 3. 返回结果 (为了兼容 TGN 原始接口的解包操作，返回 4 个值)
  # 这样 TGN 里的 self.mean_time_shift_src 和 self.mean_time_shift_dst 都会被赋值为同一个全局均值
  return mean_time_shift, std_time_shift