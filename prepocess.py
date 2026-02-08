import pandas as pd

def preprocess(file_path, delimiter=',', ):
    link_stream = pd.read_csv(file_path, delimiter=delimiter, names=['source', 'destination', 'timestamp'], index_col=False, skiprows=1)
    nodes = pd.concat([link_stream['source'], link_stream['destination']]).unique()
    print(len(nodes), "nodes in the link stream")
    node2id = {node: idx for idx, node in enumerate(nodes)}
    link_stream['source'] = link_stream['source'].map(node2id)
    link_stream['destination'] = link_stream['destination'].map(node2id)
    link_stream['idx'] = range(len(link_stream))
    output_path = file_path[:-4] + '_pcs.csv'
    link_stream.to_csv(output_path, index=False)

    return node2id