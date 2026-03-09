import numpy as np
import random
import pandas as pd
import os

class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, timestamp_norm):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.timestamp_norm = timestamp_norm
    
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)


def get_data(dataset_name, filepath, node_embedding_method):
  DEFAULT_DIM = 16
  graph_df = pd.read_csv(filepath.format(dataset_name))

  sources = graph_df.source.values
  destinations = graph_df.destination.values
  edge_idxs = graph_df.idx.values
  timestamps = graph_df.timestamp.values
  timestamp_norm = graph_df.timestamp_norm.values

  max_node_idx = max(sources.max(), destinations.max())
  num_nodes = max_node_idx + 1
  num_edges = len(graph_df)

  edge_feat_path = './data/{}.npy'.format(dataset_name)
  if os.path.exists(edge_feat_path):
      print(f"Loading edge features: {edge_feat_path}")
      edge_features = np.load(edge_feat_path)
  else:
      print(f"cannot find ({edge_feat_path}), use zero-vector for edge feat (dim={DEFAULT_DIM})...")
      edge_features = np.zeros((num_edges, DEFAULT_DIM), dtype=np.float32)

  node_feat_path = './data/{}_node.npy'.format(dataset_name)
  if os.path.exists(node_feat_path):
      print(f"cannot find node feature: {node_feat_path}")
      node_features = np.load(node_feat_path)
  else:
    if node_embedding_method == "all-zero":
      print(f"cannot find node feature: {node_feat_path}), use zero vector(dim={DEFAULT_DIM})...")
      print("Use all-zero init for node embedding. ")
      node_features = np.zeros((num_nodes, DEFAULT_DIM), dtype=np.float32)
    elif node_embedding_method == "random":
       print("Use random init for node embedding. ")
       rng = np.random.default_rng(42)
       node_features = rng.standard_normal((num_nodes, DEFAULT_DIM).astype(np.float32))
    elif node_embedding_method == "one-hot":
       print("Use one-hot init for node embedding. ")
       node_features = np.eye(num_nodes, dtype=np.float32)  

    elif node_embedding_method == "ctdne":
      ctdne_feat_path = f'pretrain/{dataset_name}.npy'
      if os.path.exists(ctdne_feat_path):
        print(f"Loading CTDNE node features: {ctdne_feat_path}")
        node_features = np.load(ctdne_feat_path, allow_pickle=True).astype(np.float32)
      else:
        raise FileNotFoundError(
          f"Cannot find random walk embedding in {ctdne_feat_path}, run pretrain.py before."
        )
    else:
      raise ValueError(
          f"Unsupported node_embedding_method: {node_embedding_method}. "
          f"Expected one of {{'random', 'one-hot', 'all-zero'}}."
      )
    

  full_data = Data(sources, destinations, timestamps, edge_idxs, timestamp_norm)

  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,full_data.n_unique_nodes))
  return node_features, edge_features, full_data


def compute_time_statistics_undirected(sources, destinations, timestamps):
  last_timestamp = dict()
  all_timediffs = []

  for k in range(len(sources)):
      source_id = sources[k]
      dest_id = destinations[k]
      c_timestamp = timestamps[k]

      if source_id not in last_timestamp:
          last_timestamp[source_id] = 0

      all_timediffs.append(c_timestamp - last_timestamp[source_id])
      last_timestamp[source_id] = c_timestamp

      if dest_id not in last_timestamp:
          last_timestamp[dest_id] = 0
          
      all_timediffs.append(c_timestamp - last_timestamp[dest_id])
      last_timestamp[dest_id] = c_timestamp

  mean_time_shift = np.mean(all_timediffs)
  std_time_shift = np.std(all_timediffs)

  return mean_time_shift, std_time_shift


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
