from lago.lago import LinkStream, lago_communities
from lago.l_modularity_function import longitudinal_modularity
from networkx.algorithms.community import louvain_communities
from pathlib import Path
import networkx as nx
import csv
import pandas as pd
import time
import matlab
import matlab.engine
import os


def lago_community(link_stream_file, nb_iter, is_stream_graph=False, info_logging=False):
    '''
    :return: dynamic communities found by LAGO

    :param link_stream_file: the path of link stream file
    :param nb_iter: number of iterations
    :param info_logging: print info logging if True
    :param record_time: record using time if True
    '''
    df = pd.read_csv(link_stream_file, header=None, index_col=False,names=["source","destination","timestamp"], skiprows=1)
    time_links = df.values.tolist()
    my_linkstream = LinkStream(is_stream_graph=is_stream_graph)
    my_linkstream.add_links(time_links)

    start_time = time.perf_counter()
    dynamic_communities = lago_communities(
        my_linkstream,
        nb_iter=nb_iter, 
        )
    end_time = time.perf_counter()

    long_mod_score = longitudinal_modularity(
            my_linkstream, 
            dynamic_communities,
            lex_type="MM"
            )

    if info_logging:
        print(f"The link stream consists of {my_linkstream.nb_edges} temporal edges\
               (or time links) accross {my_linkstream.nb_nodes} nodes and \
                {my_linkstream.network_duration} time steps, \
                of which only {my_linkstream.nb_timesteps} contain activity.")
        print(f"{len(dynamic_communities)} dynamic communities have been found")
        print(f"Longitudinal Modularity score of {long_mod_score} ")
    return dynamic_communities, long_mod_score, end_time - start_time

    
def louvain(file_path: str) -> str:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path)

    required_cols = ["source", "destination", "timestamp", "source_commu", "destination_commu"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Required: {required_cols}")

    df["source"] = df["source"].astype(str)
    df["destination"] = df["destination"].astype(str)

    u = df["source"]
    v = df["destination"]
    a = u.where(u <= v, v)
    b = v.where(u <= v, u)

    edge_w = (
        pd.DataFrame({"u": a, "v": b})
        .groupby(["u", "v"], as_index=False)
        .size()
        .rename(columns={"size": "weight"})
    )

    G = nx.Graph()
    nodes = pd.Index(df["source"]).append(pd.Index(df["destination"])).unique()
    G.add_nodes_from(nodes)

    for row in edge_w.itertuples(index=False):
        G.add_edge(row.u, row.v, weight=int(row.weight))


    communities = louvain_communities(G, weight="weight", seed=42)

    node2comm = {}
    for cid, comm_nodes in enumerate(communities):
        for n in comm_nodes:
            node2comm[n] = cid

    df["source_commu"] = df["source"].map(node2comm).astype("Int64")
    df["destination_commu"] = df["destination"].map(node2comm).astype("Int64")

    out_dir = Path("result") 
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / (file_path.name)
    df.to_csv(out_path, index=False)

    return str(out_path)

