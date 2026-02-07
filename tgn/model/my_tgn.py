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


class CommunityProjector(nn.Module):
    def __init__(self, embedding_dim, num_communities, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_communities),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.mlp(x)
    
    
class LearnableDirichletPrior(nn.Module):
    def __init__(self, K: int, alpha0: float = 10.0, alpha_min: float = 1e-3):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(K))  # learn direction
        self.alpha0 = float(alpha0)
        self.alpha_min = float(alpha_min)

    def alpha(self) -> torch.Tensor:
        q = F.softmax(self.logits, dim=0)           # [K]
        return self.alpha0 * q + self.alpha_min     # [K], >0

    def forward(self, pi: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Return negative log-likelihood: -log Dir(pi | alpha).
        pi: [K] (not necessarily perfectly normalized; we normalize inside)
        """
        pi = (pi + eps) / (pi.sum() + eps)
        alpha = self.alpha()
        alpha0 = alpha.sum()

        nll = -((alpha - 1.0) * torch.log(pi.clamp_min(eps))).sum()
        nll = nll + torch.lgamma(alpha).sum() - torch.lgamma(alpha0)
        return nll

class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift=0, std_time_shift=1, mean_time_shift_dst=0, 
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False,
               # additional parameters can be added here
               num_communities=5,
               dirichlet_alpha = 1.0
               ):
    super().__init__()

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
    
    self.dirichlet_prior = LearnableDirichletPrior(K=self.num_communities,
                                                   alpha0=dirichlet_alpha,
                                                   alpha_min=1e-3)

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, edge_times, edge_idxs, n_neighbors=20):
      n_samples = len(source_nodes)
      nodes = np.concatenate([source_nodes, destination_nodes])
      timestamps = np.concatenate([edge_times, edge_times])

      memory = None
      time_diffs = None

      if self.use_memory:
        if self.memory_update_at_start:
          memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                        self.memory.messages)
        else:
          memory = self.memory.get_memory(list(range(self.n_nodes)))
          last_update = self.memory.last_update

        source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[source_nodes].long()
        source_time_diffs = ((source_time_diffs - self.mean_time_shift) / self.std_time_shift)
        destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[destination_nodes].long()
        destination_time_diffs = ((destination_time_diffs - self.mean_time_shift) / self.std_time_shift)

        time_diffs = torch.cat([source_time_diffs, destination_time_diffs], dim=0)

      node_embedding = self.embedding_module.compute_embedding(
          memory=memory,
          source_nodes=nodes,
          timestamps=timestamps,
          n_layers=self.n_layers,
          n_neighbors=n_neighbors,
          time_diffs=time_diffs
      )

      source_node_embedding = node_embedding[:n_samples]
      destination_node_embedding = node_embedding[n_samples:]
      
      if self.use_memory:
        if self.memory_update_at_start:
          self.update_memory(nodes, self.memory.messages)
          assert torch.allclose(memory[nodes], self.memory.get_memory(nodes), atol=1e-5), "Something wrong in how the memory was updated" 
          self.memory.clear_messages(nodes)
        unique_sources, source_id_to_message = self.get_raw_messages(source_nodes,
                                                                     source_node_embedding,
                                                                     destination_nodes,
                                                                     destination_node_embedding,
                                                                     edge_times, edge_idxs)
        unique_destinations, destination_id_to_message = self.get_raw_messages(destination_nodes,
                                                                             destination_node_embedding,
                                                                             source_nodes,
                                                                             source_node_embedding,
                                                                             edge_times, edge_idxs)
        if self.memory_update_at_start:
          self.memory.store_raw_messages(unique_sources, source_id_to_message)
          self.memory.store_raw_messages(unique_destinations, destination_id_to_message)
        else: 
          '''
          merged = merge_messages(source_id_to_message, destination_id_to_message)
          unique_nodes = union(unique_sources, unique_destinations)
          self.update_memory(unique_nodes, merged)
          '''
          self.update_memory(unique_sources, source_id_to_message)
          self.update_memory(unique_destinations, destination_id_to_message)

        if self.dyrep:
          source_node_embedding = memory[source_nodes]
          destination_node_embedding = memory[destination_nodes]

      return source_node_embedding, destination_node_embedding

  def compute_community_prob(self, source_nodes, destination_nodes, edge_times, edge_idxs, n_neighbors=20):
      source_node_embedding, destination_node_embedding = self.compute_temporal_embeddings(
          source_nodes, destination_nodes, edge_times, edge_idxs, n_neighbors=n_neighbors
      )
      source_community_prob = self.community_projector(source_node_embedding)
      destination_community_prob = self.community_projector(destination_node_embedding)
      return source_community_prob, destination_community_prob
  
  def update_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(nodes, messages)
    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)
    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages, timestamps=unique_timestamps)
    

  def get_updated_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(nodes, messages)

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
                                source_time_delta_encoding],dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)


    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
