import logging
import numpy as np
import torch
from collections import defaultdict

from tgn.utils.utils import MergeLayer
from tgn.modules.my_memory import Memory
from tgn.modules.message_aggregator import get_message_aggregator
from tgn.modules.message_function import get_message_function
from tgn.modules.memory_updater import get_memory_updater
from tgn.modules.embedding_module import get_embedding_module
from tgn.model.time_encoding import TimeEncode

import torch.nn as nn
import torch.nn.functional as F


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
        n_heads=2, dropout=0.1, use_memory=False,
        memory_update_at_start=True, message_dimension=100,
        memory_dimension=500, embedding_module_type="graph_attention",
        message_function="mlp",
        mean_time_shift=0, std_time_shift=1, n_neighbors=None, aggregator_type="last",
        memory_updater_type="gru",
        use_destination_embedding_in_message=False,
        use_source_embedding_in_message=False,
        dyrep=False,
        # additional parameters can be added here
        student_time_dim=32,
        num_communities=5
        ):
    super().__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    
    """
    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
    """

    node_features = node_features.astype(np.float32)
    edge_features = edge_features.astype(np.float32)

    # TGC-style:
    # student table: trainable, initialized from input node features
    self.node_emb = nn.Parameter(torch.from_numpy(node_features).to(device))

    # teacher / anchor table: fixed copy of the same input node features
    self.pre_node_emb = torch.from_numpy(node_features).to(device)
    # keep the original names used by the rest of TGN
    self.node_raw_features = self.node_emb
    self.edge_raw_features = torch.from_numpy(edge_features).to(device)
    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.student_time_dim = int(student_time_dim)
    if self.student_time_dim <= 0:
      raise ValueError("student_time_dim must be positive")
    if self.student_time_dim % 2 != 0:
      raise ValueError("student_time_dim must be even")

    # Event-level student embedding dimension = static node feature dim + event time feature dim
    self.embedding_dimension = self.n_node_features + self.student_time_dim
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep

    self.use_memory = use_memory
    self.temporal_time_encoder = TimeEncode(dimension=self.n_node_features)
    #self.student_time_encoder = TimeEncode(dimension=self.student_time_dim)
    self.memory = None

    self.mean_time_shift = mean_time_shift
    self.std_time_shift = std_time_shift

    # Per-node last seen timestamps for building event-level student time features
    self.student_last_update = torch.zeros(self.n_nodes, dtype=torch.float32, device=self.device)

    # my change
    self.num_communities = num_communities
    self.cluster_centers = nn.Parameter(
        torch.randn(num_communities, self.embedding_dimension)
    )
    nn.init.xavier_uniform_(self.cluster_centers)
    self.teacher_src_event_emb = None   # [num_events, D]
    self.teacher_dst_event_emb = None   # [num_events, D]

    self.student_src_event_emb = None   # trainable [num_events, D]
    self.student_dst_event_emb = None   # trainable [num_events, D]

    # ==================
    if self.use_memory:
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.temporal_time_encoder.dimension
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
                                                 time_encoder=self.temporal_time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors)

    # ---- masked destination reconstruction modules ----
    self.mask_dst_token = nn.Parameter(torch.randn(1, self.embedding_dimension))
    self.mask_mlp = nn.Sequential(
        nn.Linear(self.n_node_features + self.temporal_time_encoder.dimension + self.embedding_dimension,
                  self.embedding_dimension),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(self.embedding_dimension, self.embedding_dimension),
    )
    self.dst_decoder = nn.Embedding(self.n_nodes, self.embedding_dimension)
    nn.init.xavier_uniform_(self.dst_decoder.weight)


  def compute_temporal_embeddings(
      self,
      source_nodes,
      destination_nodes,
      negative_nodes,
      edge_times,
      edge_idxs,
      n_neighbors=20
  ):
      """
      Supports:
        - negative_nodes shape [B]  OR  [B, R]
      Returns:
        - source_node_embedding: [B, D]
        - destination_node_embedding: [B, D]
        - negative_node_embedding: [B, D] or [B, R, D] (same layout as negative_nodes)
      """
      # ---------- to numpy ----------
      source_nodes = np.asarray(source_nodes, dtype=np.int64)
      destination_nodes = np.asarray(destination_nodes, dtype=np.int64)
      edge_times = np.asarray(edge_times, dtype=np.float64)
      edge_idxs = np.asarray(edge_idxs, dtype=np.int64)

      B = len(source_nodes)
      assert len(destination_nodes) == B
      assert len(edge_times) == B
      assert len(edge_idxs) == B

      # ---------- handle negative_nodes shape ----------
      neg_np = np.asarray(negative_nodes, dtype=np.int64)
      if neg_np.ndim == 1:
          # [B]
          assert len(neg_np) == B
          R = 1
          neg_flat = neg_np
          neg_timestamps = edge_times  # [B]
      elif neg_np.ndim == 2:
          # [B, R]
          assert neg_np.shape[0] == B
          R = int(neg_np.shape[1])
          neg_flat = neg_np.reshape(-1)                         # [B*R]
          neg_timestamps = np.repeat(edge_times, R)             # [B*R], each neg uses its edge time
      else:
          raise ValueError(f"negative_nodes must be 1D or 2D, got shape={neg_np.shape}")

      # concatenate nodes & timestamps for embedding_module
      nodes = np.concatenate([source_nodes, destination_nodes, neg_flat], axis=0)
      timestamps = np.concatenate([edge_times, edge_times, neg_timestamps], axis=0)

      memory = None
      time_diffs = None

      if self.use_memory:
          if self.memory_update_at_start:
              memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                            self.memory.messages)
          else:
              memory = self.memory.get_memory(list(range(self.n_nodes)))
              last_update = self.memory.last_update

          edge_times_t = torch.as_tensor(edge_times, dtype=torch.float32, device=self.device)  # [B]

          # time diffs: [B]
          src_last = last_update[torch.as_tensor(source_nodes, device=self.device)]
          dst_last = last_update[torch.as_tensor(destination_nodes, device=self.device)]

          source_time_diffs = edge_times_t - src_last
          destination_time_diffs = edge_times_t - dst_last

          # negative time diffs: [B] or [B*R]
          neg_flat_t = torch.as_tensor(neg_flat, dtype=torch.long, device=self.device)         # [B*R] or [B]
          neg_edge_times_t = torch.as_tensor(neg_timestamps, dtype=torch.float32, device=self.device)  # [B*R] or [B]
          neg_last = last_update[neg_flat_t]
          negative_time_diffs = neg_edge_times_t - neg_last

          # standardize (more reasonable)
          mean = float(self.mean_time_shift)
          std = float(self.std_time_shift) if float(self.std_time_shift) != 0 else 1.0
          source_time_diffs = (source_time_diffs - mean) / std
          destination_time_diffs = (destination_time_diffs - mean) / std
          negative_time_diffs = (negative_time_diffs - mean) / std

          # concat to match nodes order: [B + B + (B*R)]
          time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs], dim=0)

      node_embedding = self.embedding_module.compute_embedding(
          memory=memory,
          source_nodes=nodes,
          timestamps=timestamps,
          n_layers=self.n_layers,
          n_neighbors=n_neighbors,
          time_diffs=time_diffs
      )

      source_node_embedding = node_embedding[:B]                       # [B,D]
      destination_node_embedding = node_embedding[B:2 * B]            # [B,D]
      negative_node_embedding_flat = node_embedding[2 * B:]           # [B*R,D] or [B,D]

      if R > 1:
          negative_node_embedding = negative_node_embedding_flat.view(B, R, -1)  # [B,R,D]
      else:
          negative_node_embedding = negative_node_embedding_flat                # [B,D]

      # ---- memory update (unchanged logic, but dyrep needs flatten for neg) ----
      if self.use_memory:
          if self.memory_update_at_start:
              self.update_memory(nodes, self.memory.messages)
              assert torch.allclose(memory[nodes], self.memory.get_memory(nodes), atol=1e-5), \
                  "Something wrong in how the memory was updated"
              self.memory.clear_messages(nodes)

          unique_sources, source_id_to_message = self.get_raw_messages(
              source_nodes, source_node_embedding,
              destination_nodes, destination_node_embedding,
              edge_times, edge_idxs
          )
          unique_destinations, destination_id_to_message = self.get_raw_messages(
              destination_nodes, destination_node_embedding,
              source_nodes, source_node_embedding,
              edge_times, edge_idxs
          )

          if self.memory_update_at_start:
              self.memory.store_raw_messages(unique_sources, source_id_to_message)
              self.memory.store_raw_messages(unique_destinations, destination_id_to_message)
          else:
              self.update_memory(unique_sources, source_id_to_message)
              self.update_memory(unique_destinations, destination_id_to_message)

          if self.dyrep:
              source_node_embedding = memory[torch.as_tensor(source_nodes, device=self.device)]
              destination_node_embedding = memory[torch.as_tensor(destination_nodes, device=self.device)]
              neg_mem_flat = memory[torch.as_tensor(neg_flat, device=self.device)]
              negative_node_embedding = neg_mem_flat.view(B, R, -1) if R > 1 else neg_mem_flat

      return source_node_embedding, destination_node_embedding, negative_node_embedding


  def compute_masked_dst_representations(self, source_nodes, edge_times, edge_idxs=None, n_neighbors=20):
      """
      Build masked-event representations for destination reconstruction.

      source_nodes: np.ndarray or list, shape [B]
      edge_times: np.ndarray or list, shape [B]
      edge_idxs: optional, kept for interface consistency
      n_neighbors: int

      Returns
      -------
      h_mask: torch.Tensor of shape [B, D]
      """
      source_nodes = np.asarray(source_nodes, dtype=np.int64)
      edge_times = np.asarray(edge_times, dtype=np.float64)
      B = len(source_nodes)

      memory = None
      time_diffs = None

      if self.use_memory:
          if self.memory_update_at_start:
              memory, last_update = self.get_updated_memory(list(range(self.n_nodes)), self.memory.messages)
          else:
              memory = self.memory.get_memory(list(range(self.n_nodes)))
              last_update = self.memory.last_update

          edge_times_t = torch.as_tensor(edge_times, dtype=torch.float32, device=self.device)
          src_last = last_update[torch.as_tensor(source_nodes, dtype=torch.long, device=self.device)]
          source_time_diffs = edge_times_t - src_last

          mean = float(self.mean_time_shift)
          std = float(self.std_time_shift) if float(self.std_time_shift) != 0 else 1.0
          time_diffs = (source_time_diffs - mean) / std

      source_embedding = self.embedding_module.compute_embedding(
          memory=memory,
          source_nodes=source_nodes,
          timestamps=edge_times,
          n_layers=self.n_layers,
          n_neighbors=n_neighbors,
          time_diffs=time_diffs,
      )

      delta_t = torch.zeros((B, 1), dtype=torch.float32, device=self.device)
      time_emb = self.temporal_time_encoder(delta_t).view(B, -1)
      mask_token = self.mask_dst_token.expand(B, -1)

      h_mask = self.mask_mlp(torch.cat([source_embedding, time_emb, mask_token], dim=1))
      return h_mask

  def compute_masked_destination_scores(self, source_nodes, destination_nodes,
                                        negative_destination_nodes, edge_times,
                                        edge_idxs=None, n_neighbors=20):
    """
    Compute scores for masked-destination reconstruction.

    source_nodes: [B]
    destination_nodes: [B]
    negative_destination_nodes: [B] or [B, R]
    edge_times: [B]

    Returns
    -------
    pos_score: [B]
    neg_score: [B] or [B, R]
    """
    source_nodes = np.asarray(source_nodes, dtype=np.int64)
    destination_nodes = np.asarray(destination_nodes, dtype=np.int64)
    negative_destination_nodes = np.asarray(negative_destination_nodes, dtype=np.int64)
    edge_times = np.asarray(edge_times, dtype=np.float64)

    h_mask = self.compute_masked_dst_representations(
        source_nodes=source_nodes,
        edge_times=edge_times,
        edge_idxs=edge_idxs,
        n_neighbors=n_neighbors,
    )  # [B, D]

    pos_dst_t = torch.as_tensor(destination_nodes, dtype=torch.long, device=self.device)
    pos_dst_emb = self.dst_decoder(pos_dst_t)  # [B, D]
    pos_score = torch.sum(h_mask * pos_dst_emb, dim=-1)  # [B]

    if negative_destination_nodes.ndim == 1:
        neg_dst_t = torch.as_tensor(negative_destination_nodes, dtype=torch.long, device=self.device)
        neg_dst_emb = self.dst_decoder(neg_dst_t)  # [B, D]
        neg_score = torch.sum(h_mask * neg_dst_emb, dim=-1)  # [B]
    elif negative_destination_nodes.ndim == 2:
        neg_dst_t = torch.as_tensor(negative_destination_nodes, dtype=torch.long, device=self.device)
        neg_dst_emb = self.dst_decoder(neg_dst_t)  # [B, R, D]
        neg_score = torch.sum(h_mask.unsqueeze(1) * neg_dst_emb, dim=-1)  # [B, R]
    else:
        raise ValueError(f"negative_destination_nodes must be 1D or 2D, got shape={negative_destination_nodes.shape}")

    return pos_score, neg_score


  def compute_masked_destination_loss(self, source_nodes, destination_nodes,
                                      negative_destination_nodes, edge_times,
                                      edge_idxs=None, n_neighbors=20):
      """
      Logistic negative-sampling loss for masked destination reconstruction.
      """
      pos_score, neg_score = self.compute_masked_destination_scores(
          source_nodes=source_nodes,
          destination_nodes=destination_nodes,
          negative_destination_nodes=negative_destination_nodes,
          edge_times=edge_times,
          edge_idxs=edge_idxs,
          n_neighbors=n_neighbors,
      )

      pos_loss = -F.logsigmoid(pos_score)
      if neg_score.dim() == 1:
          neg_loss = -F.logsigmoid(-neg_score)
      else:
          neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)

      return (pos_loss + neg_loss).mean()

  def compute_node_event_embeddings_for_clustering(
    self,
    source_nodes,
    destination_nodes,
    edge_times,
    edge_idxs,
    n_neighbors=20
    ):
    """
    Scheme B:
    directly use a trainable event-embedding table initialized from teacher.
    """
    if self.student_src_event_emb is None or self.student_dst_event_emb is None:
        raise ValueError(
            "Student event embeddings are not initialized. "
            "Call load_teacher_node_event_embeddings(...) first."
        )

    edge_idxs = np.asarray(edge_idxs, dtype=np.int64)
    edge_idx_t = torch.as_tensor(edge_idxs, dtype=torch.long, device=self.device)

    src_emb = self.student_src_event_emb[edge_idx_t]  # [B, D]
    dst_emb = self.student_dst_event_emb[edge_idx_t]  # [B, D]

    z_batch = torch.cat([src_emb, dst_emb], dim=0)  # [2B, D]
    return z_batch    
  

  def load_teacher_node_event_embeddings(self, src_npy_path, dst_npy_path):
    src = np.load(src_npy_path).astype(np.float32)   # [num_events, D]
    dst = np.load(dst_npy_path).astype(np.float32)   # [num_events, D]

    if src.shape[1] != self.embedding_dimension:
        raise ValueError(
            f"Teacher src dim {src.shape[1]} != model embedding dim {self.embedding_dimension}"
        )
    if dst.shape[1] != self.embedding_dimension:
        raise ValueError(
            f"Teacher dst dim {dst.shape[1]} != model embedding dim {self.embedding_dimension}"
        )

    # fixed teacher tables
    self.teacher_src_event_emb = torch.from_numpy(src).to(self.device)
    self.teacher_dst_event_emb = torch.from_numpy(dst).to(self.device)

    # Scheme B:
    # trainable student tables initialized by directly copying teacher
    self.student_src_event_emb = nn.Parameter(self.teacher_src_event_emb.clone())
    self.student_dst_event_emb = nn.Parameter(self.teacher_dst_event_emb.clone())
    

  def get_teacher_node_event_embeddings_for_batch(self, edge_idxs):
    if self.teacher_src_event_emb is None or self.teacher_dst_event_emb is None:
        raise ValueError("Teacher node-event embeddings are not loaded.")

    edge_idxs = np.asarray(edge_idxs, dtype=np.int64)
    edge_idx_t = torch.as_tensor(edge_idxs, dtype=torch.long, device=self.device)

    src_teacher = self.teacher_src_event_emb[edge_idx_t]  # [B, D]
    dst_teacher = self.teacher_dst_event_emb[edge_idx_t]  # [B, D]

    z_teacher = torch.cat([src_teacher, dst_teacher], dim=0)  # [2B, D]
    return z_teacher
  

  def student_t_assignment(self, z, alpha=1.0):
    """
    z: [M, D]
    return q: [M, K]
    """
    dist_sq = torch.sum(
        (z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2,
        dim=2
    )  # [M, K]

    q = 1.0 / (1.0 + dist_sq / alpha)
    q = q ** ((alpha + 1.0) / 2.0)
    q = q / torch.sum(q, dim=1, keepdim=True)
    return q
  
  def target_distribution(self, q):
    weight = (q ** 2) / torch.sum(q, dim=0, keepdim=True)
    p = weight / torch.sum(weight, dim=1, keepdim=True)
    return p
  
  def compute_node_event_cluster_loss(
    self,
    source_nodes,
    destination_nodes,
    edge_times,
    edge_idxs,
    n_neighbors=20,
    alpha=1.0
):
    z_batch = self.compute_node_event_embeddings_for_clustering(
        source_nodes=source_nodes,
        destination_nodes=destination_nodes,
        edge_times=edge_times,
        edge_idxs=edge_idxs,
        n_neighbors=n_neighbors
    )   # [2B, D]

    # current model assignment
    q = self.student_t_assignment(z_batch, alpha=alpha)  # [2B, K]

    # teacher assignment from fixed CTDNE+time embeddings
    with torch.no_grad():
        z_teacher = self.get_teacher_node_event_embeddings_for_batch(edge_idxs)  # [2B, D]
        q_teacher = self.student_t_assignment(z_teacher, alpha=alpha)            # [2B, K]
        p = self.target_distribution(q_teacher)                                  # [2B, K]

    loss = F.kl_div(torch.log(q + 1e-12), p, reduction="batchmean")

    # update per-node last seen timestamps after using current batch deltas
    source_nodes = np.asarray(source_nodes, dtype=np.int64)
    destination_nodes = np.asarray(destination_nodes, dtype=np.int64)
    edge_times = np.asarray(edge_times, dtype=np.float32)

    src_t = torch.as_tensor(source_nodes, dtype=torch.long, device=self.device)
    dst_t = torch.as_tensor(destination_nodes, dtype=torch.long, device=self.device)
    edge_times_t = torch.as_tensor(edge_times, dtype=torch.float32, device=self.device)

    with torch.no_grad():
        self.student_last_update[src_t] = edge_times_t
        self.student_last_update[dst_t] = edge_times_t

    return loss, q
  


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
    source_time_delta_encoding = self.temporal_time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages
  def fixed_time_encoding_batch_torch(self, z_values, d_model):
    """
    Fixed sin-cos time encoding, matched to the preprocessing function
    used to build teacher node-event features.
    """
    if d_model % 2 != 0:
        raise ValueError("d_model must be even")

    z_values = z_values.float().view(-1)  # [B]
    pe = torch.zeros((z_values.shape[0], d_model),
                    dtype=torch.float32,
                    device=z_values.device)

    half = d_model // 2
    i = torch.arange(half, dtype=torch.float32, device=z_values.device)
    denom = torch.pow(
        torch.tensor(10000.0, dtype=torch.float32, device=z_values.device),
        2 * i / d_model
    )  # [half]

    pe[:, 0::2] = torch.sin(z_values.unsqueeze(1) / denom.unsqueeze(0))
    pe[:, 1::2] = torch.cos(z_values.unsqueeze(1) / denom.unsqueeze(0))
    return pe

  def reset_student_last_update(self):
    self.student_last_update.zero_()
  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
