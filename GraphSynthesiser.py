from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
import multilayerGM as gm


@dataclass
class GraphSynthesiser:
    lam: float = 5.0
    seed: Optional[int] = 42
    base_time: int = 0
    sort_by_time: bool = True
    make_strictly_increasing: bool = True
    layer_index: int = 1
    enforce_same_layer: bool = True

    # cached
    dt: Any = None
    null: Any = None
    partition: Any = None
    multinet: Optional[nx.Graph] = None

    def build_dependency_tensor(self, n_nodes: int, n_layers: int, p: float) -> Any:
        """Create and cache dependency tensor."""
        self.dt = gm.dependency_tensors.UniformMultiplex(n_nodes, n_layers, p)
        return self.dt

    def build_null(self, theta: float, n_sets: int) -> Any:
        """Create and cache Dirichlet null given an existing dependency tensor."""
        if self.dt is None:
            raise ValueError("dependency tensor (dt) is None. Call build_dependency_tensor(...) first.")
        self.null = gm.DirichletNull(layers=self.dt.shape[1:], theta=theta, n_sets=n_sets)
        return self.null

    def sample_partition(self) -> Any:
        """Sample and cache a partition given existing dt and null."""
        if self.dt is None:
            raise ValueError("dependency tensor (dt) is None. Call build_dependency_tensor(...) first.")
        if self.null is None:
            raise ValueError("null distribution is None. Call build_null(...) first.")
        self.partition = gm.sample_partition(dependency_tensor=self.dt, null_distribution=self.null)
        return self.partition

    def generate_multilayer_network(self, mu: float, k_min: int, k_max: int, t_k: float) -> nx.Graph:
        """Generate and cache a multilayer DCSBM network given an existing partition."""
        if self.partition is None:
            raise ValueError("partition is None. Call sample_partition() first.")
        self.multinet = gm.multilayer_DCSBM_network(
            self.partition, mu=mu, k_min=k_min, k_max=k_max, t_k=t_k
        )
        return self.multinet


    def save_graph_csv_poisson_int_ts(self, G: nx.Graph, out_csv: str, **overrides) -> None:
        lam = overrides.get("lam", self.lam)
        seed = overrides.get("seed", self.seed)
        base_time = overrides.get("base_time", self.base_time)
        sort_by_time = overrides.get("sort_by_time", self.sort_by_time)
        make_strictly_increasing = overrides.get("make_strictly_increasing", self.make_strictly_increasing)
        layer_index = overrides.get("layer_index", self.layer_index)
        enforce_same_layer = overrides.get("enforce_same_layer", self.enforce_same_layer)

        rng = np.random.default_rng(seed)

        edges = list(G.edges(keys=True)) if G.is_multigraph() else list(G.edges())
        edges = [(u, v) for (u, v, *rest) in edges]

        df_cols = ["source", "destination", "timestamp", "source_commu", "destination_commu"]
        if len(edges) == 0:
            pd.DataFrame(columns=df_cols).to_csv(out_csv, index=False)
            return

        edges_by_layer: dict[Any, list[tuple[Any, Any]]] = {}
        for (u, v) in edges:
            lu = u[layer_index] if isinstance(u, tuple) and len(u) > layer_index else None
            lv = v[layer_index] if isinstance(v, tuple) and len(v) > layer_index else None

            if enforce_same_layer and (lu != lv):
                raise ValueError(f"Edge endpoints are in different layers: {u} vs {v}")

            edges_by_layer.setdefault(lu, []).append((u, v))

        shuffled_edges: list[tuple[Any, Any]] = []
        for layer in sorted(edges_by_layer.keys(), key=lambda x: (x is None, x)):
            layer_edges = edges_by_layer[layer]
            rng.shuffle(layer_edges)
            shuffled_edges.extend(layer_edges)

        ts = rng.poisson(lam=lam, size=len(shuffled_edges)).astype(np.int64)
        if make_strictly_increasing:
            ts = np.cumsum(ts + 1).astype(np.int64)
        ts = (ts + np.int64(base_time)).astype(np.int64)

        rows = []
        for i, (u, v) in enumerate(shuffled_edges):
            rows.append({
                "source": u[0] if isinstance(u, tuple) else u,
                "destination": v[0] if isinstance(v, tuple) else v,
                "timestamp": int(ts[i]),
                "source_commu": G.nodes[u].get("mesoset", None),
                "destination_commu": G.nodes[v].get("mesoset", None),
            })

        df = pd.DataFrame(rows)
        df["timestamp"] = df["timestamp"].astype("int64")
        if sort_by_time:
            df = df.sort_values(["timestamp"], kind="stable").reset_index(drop=True)
        df.to_csv(out_csv, index=False)

    def synthesize_to_csv(
        self,
        out_csv: str,
        *,
        n_nodes: int,
        n_layers: int,
        p: float,
        theta: float,
        n_sets: int,
        mu: float,
        k_min: int,
        k_max: int,
        t_k: float,
        **export_overrides,
    ) -> nx.Graph:
        """End-to-end synthesis to CSV with parameters provided at call time."""
        self.build_dependency_tensor(n_nodes=n_nodes, n_layers=n_layers, p=p)
        self.build_null(theta=theta, n_sets=n_sets)
        self.sample_partition()
        G = self.generate_multilayer_network(mu=mu, k_min=k_min, k_max=k_max, t_k=t_k)
        self.save_graph_csv_poisson_int_ts(G, out_csv=out_csv, **export_overrides)
        return G