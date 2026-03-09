import pandas as pd

def preprocess(file_path, delimiter=',', time_scale=None):
    link_stream = pd.read_csv(
        file_path,
        delimiter=delimiter,
        names=['source', 'destination', 'timestamp'],
        index_col=False,
        skiprows=1
    )

    # map nodes -> ids
    nodes = pd.concat([link_stream['source'], link_stream['destination']]).unique()
    print(len(nodes), "nodes in the link stream")
    node2id = {node: idx for idx, node in enumerate(nodes)}
    link_stream['source'] = link_stream['source'].map(node2id)
    link_stream['destination'] = link_stream['destination'].map(node2id)


    link_stream['idx'] = range(len(link_stream))

    t = pd.to_numeric(link_stream['timestamp'], errors='coerce')
    if t.isna().any():
        raise ValueError("timestamp column contains non-numeric values after parsing.")
    t_min = float(t.min())
    if time_scale is None:
        span = float(t.max() - t_min)
        time_scale = span / 1.0 if span > 0 else 1.0

    link_stream['timestamp_norm'] = (t - t_min) / float(time_scale)

    output_path = file_path[:-4] + '_pcs.csv'
    link_stream.to_csv(output_path, index=False)

    return node2id, len(nodes)