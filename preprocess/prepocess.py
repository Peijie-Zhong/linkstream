import os
import json
import pandas as pd


def preprocess(file_path, delimiter=",", output_dir=None):
    if output_dir is None:
        output_dir = "preprocess"

    os.makedirs(output_dir, exist_ok=True)

    link_stream = pd.read_csv(
        file_path,
        delimiter=delimiter,
        usecols=[0, 1, 2]
    )
    link_stream.columns = ["source", "destination", "timestamp"]

    nodes = pd.concat([link_stream["source"], link_stream["destination"]]).unique()
    print(f"{len(nodes)} nodes in the link stream")

    node2id = {int(node): int(idx) for idx, node in enumerate(nodes)}

    link_stream["source"] = link_stream["source"].map(node2id)
    link_stream["destination"] = link_stream["destination"].map(node2id)
    link_stream["idx"] = range(len(link_stream))

    t = pd.to_numeric(link_stream["timestamp"], errors="coerce")
    if t.isna().any():
        raise ValueError(f"{file_path} contains non-numeric timestamps.")

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    output_csv = os.path.join(output_dir, base_name + ".csv")
    output_map_csv = os.path.join(output_dir, base_name + "_node2id.csv")

    link_stream.to_csv(output_csv, index=False)

    map_df = pd.DataFrame({
        "node": list(node2id.keys()),
        "id": list(node2id.values())
    })
    map_df.to_csv(output_map_csv, index=False)

    return node2id, len(nodes)