import os
import glob
import pandas as pd
from tgn.utils.my_data import get_data  
import importlib
import sys
importlib.reload(sys.modules['tgn.utils.my_data'])
from tgn.utils.utils import get_neighbor_finder
importlib.reload(sys.modules['tgn.utils.utils'])
from tgn.utils.my_data import get_data, compute_time_statistics, compute_time_statistics_undirected
from tgn.utils.my_data import get_data  
import numpy as np
import torch
from sklearn.cluster import KMeans


def init_cluster_centers_from_saved_node_embeddings(
    tgn,
    num_clusters,
    node_npy_path="/Users/acw721/Desktop/research/linkstream/pretrain/p0.8_mu0.2_1.npy",
    max_samples=None,
    random_state=0,
):
    node_emb = np.load(node_npy_path)   # [N, D]

    if node_emb.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={node_emb.shape}")

    X = node_emb

    if max_samples is not None and len(X) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]

    feat_dim = X.shape[1]
    model_dim = tgn.embedding_dimension

    if feat_dim != model_dim:
        raise ValueError(
            f"Saved node embedding dim ({feat_dim}) != tgn.embedding_dimension ({model_dim})."
        )

    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init=10,
        random_state=random_state
    )
    kmeans.fit(X)

    centers = torch.tensor(
        kmeans.cluster_centers_,
        dtype=torch.float32,
        device=tgn.cluster_centers.device
    )

    with torch.no_grad():
        tgn.cluster_centers.copy_(centers)

    print("Initialized cluster centers from saved node-level embeddings")
    print("node_emb shape:", node_emb.shape)
    print("kmeans centers shape:", kmeans.cluster_centers_.shape)


import copy
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from evaluation import dynamic_mi


def evaluate_epoch_ami_in_memory(
    tgn,
    data,
    gt_partition,
    batch_size,
    n_neighbors,
    num_clusters,
    use_memory=True,
    average_method="arithmetic",
    normalisation="ami",
):
    tgn.eval()

    src_emb_list = []
    dst_emb_list = []

    all_sources = []
    all_destinations = []
    all_timestamps = []

    num_instance = len(data.sources)
    num_batch = (num_instance + batch_size - 1) // batch_size

    if use_memory and tgn.use_memory:
        tgn.memory.__init_memory__()
    with torch.no_grad():
        for k in range(num_batch):
            start_idx = k * batch_size
            end_idx = min(num_instance, start_idx + batch_size)

            sources_batch = data.sources[start_idx:end_idx]
            destinations_batch = data.destinations[start_idx:end_idx]
            edge_idxs_batch = data.edge_idxs[start_idx:end_idx]
            timestamps_batch = data.timestamps[start_idx:end_idx]


            src_emb, dst_emb = tgn.compute_src_dst_embeddings(
                source_nodes=sources_batch,
                destination_nodes=destinations_batch,
                edge_times=timestamps_batch,
                edge_idxs=edge_idxs_batch,
                n_neighbors=n_neighbors
            )

            src_emb_list.append(src_emb.detach().cpu().numpy())
            dst_emb_list.append(dst_emb.detach().cpu().numpy())

            all_sources.append(np.asarray(sources_batch))
            all_destinations.append(np.asarray(destinations_batch))
            all_timestamps.append(np.asarray(timestamps_batch))

            if use_memory and tgn.use_memory:
                tgn.memory.detach_memory()

    src_emb_all = np.concatenate(src_emb_list, axis=0)
    dst_emb_all = np.concatenate(dst_emb_list, axis=0)

    sources_all = np.concatenate(all_sources, axis=0)
    destinations_all = np.concatenate(all_destinations, axis=0)
    timestamps_all = np.concatenate(all_timestamps, axis=0)

    # node-event clustering
    X = np.concatenate([src_emb_all, dst_emb_all], axis=0)
    labels = KMeans(
        n_clusters=num_clusters,
        n_init=10,
        random_state=0
    ).fit_predict(X)

    N = len(sources_all)
    src_labels = labels[:N]
    dst_labels = labels[N:]

    pred_partition = build_partition_from_arrays(
        sources=sources_all,
        destinations=destinations_all,
        timestamps=timestamps_all,
        source_labels=src_labels,
        destination_labels=dst_labels,
        on_conflict="keep_last",
    )

    score = dynamic_mi(
        gt=gt_partition,
        pred=pred_partition,
        average_method=average_method,
        normalisation=normalisation,
    )
    return score



node2id_df = pd.read_csv("preprocess/p0.8_mu0.2_1_node2id.csv")
node2id = dict(zip(node2id_df["node"], node2id_df["id"]))
link_stream = pd.read_csv('preprocess/p0.8_mu0.2_1.csv')


data = 'p0.8_mu0.2_1'

node_feat, edge_feat, full_data = get_data(data, "preprocess/p0.8_mu0.2_1.csv", node_embedding_method="ctdne", node2id=node2id)

max_idx = max(full_data.unique_nodes)


ngh_finder = get_neighbor_finder(full_data, uniform=True, max_node_idx=max_idx)
mean_time_shift, std_time_shift= compute_time_statistics_undirected(full_data.sources, 
                                full_data.destinations, 
                                full_data.timestamps)


import importlib
import tgn.model.my_tgn as my_tgn
importlib.reload(my_tgn)
TGN = my_tgn.TGN
from pathlib import Path


import logging
import time
from pathlib import Path
import torch

NUM_EPOCH = 20
BATCH_SIZE = 128
NUM_NEIGHBORS = 20
NUM_HEADS = 4
DROP_OUT = 0.1
NUM_LAYER = 2
LEARNING_RATE = 0.01
NODE_DIM = node_feat.shape[1]
TIME_DIM = 128
USE_MEMORY = False
MESSAGE_DIM = 128
#MEMORY_DIM = NODE_DIM
MEMORY_DIM = 128
num_communities = 5
device = 'cuda' if torch.cuda.is_available() else 'mps'
prefix = 'syn_net'
aggregator = 'mean'
memory_update_at_end = False
embedding_module = 'graph_attention' # graph_attention, graph_sum, time, identity
message_function = 'mlp'

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{prefix}-{data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{prefix}-{data}-{epoch}.pth'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

tgn = TGN(
    neighbor_finder=ngh_finder,
    node_features=node_feat,
    edge_features=edge_feat,
    device=device,
    n_layers=NUM_LAYER,
    n_heads=NUM_HEADS,
    dropout=DROP_OUT,
    use_memory=USE_MEMORY,
    message_dimension=MESSAGE_DIM,
    memory_dimension=MEMORY_DIM,
    n_neighbors=NUM_NEIGHBORS,
    mean_time_shift=mean_time_shift,
    std_time_shift=std_time_shift,
    use_destination_embedding_in_message=True,
    use_source_embedding_in_message=True,
    memory_update_at_start=not memory_update_at_end,
    embedding_module_type=embedding_module,
    message_function=message_function,
    aggregator_type=aggregator,
).to(device)


init_cluster_centers_from_saved_node_embeddings(
    tgn=tgn,
    num_clusters=num_communities,
    node_npy_path="/Users/acw721/Desktop/research/linkstream/pretrain/p0.8_mu0.2_1.npy",
    max_samples=None,
    random_state=0,
)


optimizer = torch.optim.AdamW(tgn.parameters(), lr=LEARNING_RATE)


import math
num_instance = len(full_data.sources)
num_batches = math.ceil(len(full_data.sources) / BATCH_SIZE)
print(f'num_batches: {num_batches}')

NUM_NEG = 2

def sample_negative_destinations(num_nodes, positive_destinations, num_neg):
  positive_destinations = np.asarray(positive_destinations, dtype=np.int64)
  batch_size = len(positive_destinations)
  neg_dst = np.random.randint(0, num_nodes, size=(batch_size, num_neg), dtype=np.int64)

  for i in range(batch_size):
    for j in range(num_neg):
      while neg_dst[i, j] == positive_destinations[i]:
        neg_dst[i, j] = np.random.randint(0, num_nodes)

  return neg_dst


def build_partition_from_arrays(
    sources,
    destinations,
    timestamps,
    source_labels,
    destination_labels,
    on_conflict="keep_last",
):
    part = {}

    def _assign(k, v):
        if k not in part:
            part[k] = v
            return
        if part[k] == v:
            return
        if on_conflict == "keep_first":
            return
        if on_conflict == "keep_last":
            part[k] = v
            return
        raise ValueError(f"Conflict on {k}: existing={part[k]} new={v}")

    for s, d, t, cs, cd in zip(
        sources, destinations, timestamps, source_labels, destination_labels
    ):
        _assign((int(s), int(t)), int(cs))
        _assign((int(d), int(t)), int(cd))

    return part

from typing import Dict, Hashable, Tuple, Any, Optional, Literal
import pandas as pd
Node = Hashable
Time = Hashable
Key = Tuple[Node, Time]

def build_partition_from_csv(
    csv_path: str,
    *,
    source_col: str = "source",
    destination_col: str = "destination",
    timestamp_col: str = "timestamp",
    source_commu_col: str = "source_commu",
    destination_commu_col: str = "destination_commu",
    sep: str = ",",
    header: Optional[int] = "infer",
    skip_first_row: bool = False,
    dtype_source: Any = int,
    dtype_destination: Any = int,
    dtype_timestamp: Any = int,
    dtype_commu: Any = int,
    on_conflict: Literal["keep_first", "keep_last", "error"] = "keep_last",
    node2id: Optional[Dict[Any, int]] = None,
) -> Dict[Key, Any]:
    usecols = [source_col, destination_col, timestamp_col, source_commu_col, destination_commu_col]

    df = pd.read_csv(
        csv_path,
        sep=sep,
        header=header,
        usecols=usecols,
        skiprows=1 if skip_first_row else None,
    )

    df[source_col] = df[source_col].astype(dtype_source)
    df[destination_col] = df[destination_col].astype(dtype_destination)
    df[timestamp_col] = df[timestamp_col].astype(dtype_timestamp)
    df[source_commu_col] = df[source_commu_col].astype(dtype_commu)
    df[destination_commu_col] = df[destination_commu_col].astype(dtype_commu)

    part: Dict[Key, Any] = {}

    def _map_node(x):
        if node2id is None:
            return x
        if x not in node2id:
            raise KeyError(f"node {x} not found in node2id")
        return node2id[x]

    def _assign(k: Key, v: Any):
        if k not in part:
            part[k] = v
            return
        if part[k] == v:
            return
        if on_conflict == "keep_first":
            return
        if on_conflict == "keep_last":
            part[k] = v
            return
        raise ValueError(f"Conflict on {k}: existing={part[k]} new={v}")

    for s, d, t, cs, cd in df[
        [source_col, destination_col, timestamp_col, source_commu_col, destination_commu_col]
    ].itertuples(index=False, name=None):
        s = _map_node(s)
        d = _map_node(d)
        _assign((s, t), cs)
        _assign((d, t), cd)

    return part


def load_dataset_bundle(data_name, preprocess_dir="preprocess", pretrain_dir="pretrain"):
    node2id_path = os.path.join(preprocess_dir, f"{data_name}_node2id.csv")
    processed_csv_path = os.path.join(preprocess_dir, f"{data_name}.csv")
    node_npy_path = os.path.join(pretrain_dir, f"{data_name}.npy")
    raw_csv_path = os.path.join("syn_data", f"{data_name}.csv")

    required_paths = {
        "node2id_csv": node2id_path,
        "processed_csv": processed_csv_path,
        "node_npy": node_npy_path,
        "raw_csv": raw_csv_path,
    }
    for key, path in required_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {key} for {data_name}: {path}")

    node2id_df = pd.read_csv(node2id_path)
    node2id = {int(k): int(v) for k, v in zip(node2id_df["node"], node2id_df["id"])}

    node_feat, edge_feat, full_data = get_data(
        data_name,
        processed_csv_path,
        node_embedding_method="ctdne",
        node2id=node2id,
    )

    gt_partition = build_partition_from_csv(
        raw_csv_path,
        node2id=node2id,
    )

    return {
        "data_name": data_name,
        "node2id": node2id,
        "node_feat": node_feat,
        "edge_feat": edge_feat,
        "full_data": full_data,
        "gt_partition": gt_partition,
        "processed_csv_path": processed_csv_path,
        "node_npy_path": node_npy_path,
        "raw_csv_path": raw_csv_path,
    }


import logging
import time
from pathlib import Path
import torch

NUM_EPOCH = 20
BATCH_SIZE = 128
NUM_NEIGHBORS = 20
NUM_HEADS = 4
DROP_OUT = 0.1
NUM_LAYER = 2
LEARNING_RATE = 0.01
TIME_DIM = 128
USE_MEMORY = False
MESSAGE_DIM = 128
MEMORY_DIM = 128
num_communities = 5
device = 'cuda' if torch.cuda.is_available() else 'mps'
prefix = 'syn_net'
aggregator = 'mean'
memory_update_at_end = False
embedding_module = 'graph_attention' # graph_attention, graph_sum, time, identity
message_function = 'mlp'

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
Path("log/").mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)


import math

NUM_NEG = 2


def sample_negative_destinations(num_nodes, positive_destinations, num_neg):
    positive_destinations = np.asarray(positive_destinations, dtype=np.int64)
    batch_size = len(positive_destinations)
    neg_dst = np.random.randint(0, num_nodes, size=(batch_size, num_neg), dtype=np.int64)

    for i in range(batch_size):
        for j in range(num_neg):
            while neg_dst[i, j] == positive_destinations[i]:
                neg_dst[i, j] = np.random.randint(0, num_nodes)

    return neg_dst


import copy
from typing import Dict, Hashable, Tuple, Any, Optional, Literal
import pandas as pd

Node = Hashable
Time = Hashable
Key = Tuple[Node, Time]


def train_one_dataset(data_name, preprocess_dir="preprocess", pretrain_dir="pretrain"):
    bundle = load_dataset_bundle(
        data_name=data_name,
        preprocess_dir=preprocess_dir,
        pretrain_dir=pretrain_dir,
    )

    node_feat = bundle["node_feat"]
    edge_feat = bundle["edge_feat"]
    full_data = bundle["full_data"]
    gt_partition = bundle["gt_partition"]
    node_npy_path = bundle["node_npy_path"]

    max_idx = max(full_data.unique_nodes)
    ngh_finder = get_neighbor_finder(full_data, uniform=True, max_node_idx=max_idx)
    mean_time_shift, std_time_shift = compute_time_statistics_undirected(
        full_data.sources,
        full_data.destinations,
        full_data.timestamps,
    )

    import importlib
    import tgn.model.my_tgn as my_tgn
    importlib.reload(my_tgn)
    TGN = my_tgn.TGN

    MODEL_SAVE_PATH = f'./saved_models/{prefix}-{data_name}.pth'
    get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{prefix}-{data_name}-{epoch}.pth'
    _ = MODEL_SAVE_PATH, get_checkpoint_path

    tgn = TGN(
        neighbor_finder=ngh_finder,
        node_features=node_feat,
        edge_features=edge_feat,
        device=device,
        n_layers=NUM_LAYER,
        n_heads=NUM_HEADS,
        dropout=DROP_OUT,
        use_memory=USE_MEMORY,
        message_dimension=MESSAGE_DIM,
        memory_dimension=MEMORY_DIM,
        n_neighbors=NUM_NEIGHBORS,
        mean_time_shift=mean_time_shift,
        std_time_shift=std_time_shift,
        use_destination_embedding_in_message=True,
        use_source_embedding_in_message=True,
        memory_update_at_start=not memory_update_at_end,
        embedding_module_type=embedding_module,
        message_function=message_function,
        aggregator_type=aggregator,
    ).to(device)

    init_cluster_centers_from_saved_node_embeddings(
        tgn=tgn,
        num_clusters=num_communities,
        node_npy_path=node_npy_path,
        max_samples=None,
        random_state=0,
    )

    optimizer = torch.optim.AdamW(tgn.parameters(), lr=LEARNING_RATE)

    num_instance = len(full_data.sources)
    num_batches = math.ceil(num_instance / BATCH_SIZE)
    print(f'[{data_name}] num_batches: {num_batches}')

    epoch_amis = []
    train_losses = []

    best_ami = -1.0
    best_epoch = -1
    best_state_dict = None

    backprop_every = 1

    local_num_epoch = 50
    for epoch in range(local_num_epoch):
        start_epoch = time.time()
        if USE_MEMORY:
            tgn.memory.__init_memory__()

        tgn.set_neighbor_finder(ngh_finder)
        m_loss = []

        logger.info(f'[{data_name}] start {epoch} epoch')

        for k in range(0, num_batches, backprop_every):
            optimizer.zero_grad()

            for j in range(backprop_every):
                batch_idx = k + j
                if batch_idx >= num_batches:
                    continue

                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)

                sources_batch = full_data.sources[start_idx:end_idx]
                destinations_batch = full_data.destinations[start_idx:end_idx]
                edge_idxs_batch = full_data.edge_idxs[start_idx:end_idx]
                timestamps_batch = full_data.timestamps[start_idx:end_idx]

                neg_dst_batch = sample_negative_destinations(
                    num_nodes=tgn.n_nodes,
                    positive_destinations=destinations_batch,
                    num_neg=NUM_NEG,
                )

                tgn.train()

                mask_loss, node_event_loss, q_batch = tgn.compute_joint_mask_and_cluster_loss(
                    source_nodes=sources_batch,
                    destination_nodes=destinations_batch,
                    negative_destination_nodes=neg_dst_batch,
                    edge_times=timestamps_batch,
                    edge_idxs=edge_idxs_batch,
                    n_neighbors=NUM_NEIGHBORS,
                    alpha=1.0,
                )
                _ = q_batch
                lambda_node = 1.0
                loss = 1.0 * mask_loss + lambda_node * node_event_loss

                loss /= backprop_every
                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())

                if USE_MEMORY:
                    tgn.memory.detach_memory()

        epoch_time = time.time() - start_epoch
        mean_train_loss = float(np.mean(m_loss)) if len(m_loss) > 0 else float('nan')
        train_losses.append(mean_train_loss)

        logger.info(f'[{data_name}] epoch: {epoch} took {epoch_time:.2f}s')
        logger.info(f'[{data_name}] Epoch mean total loss: {mean_train_loss}')

        try:
            epoch_ami = evaluate_epoch_ami_in_memory(
                tgn=tgn,
                data=full_data,
                gt_partition=gt_partition,
                batch_size=BATCH_SIZE,
                n_neighbors=NUM_NEIGHBORS,
                num_clusters=num_communities,
                use_memory=USE_MEMORY,
                average_method="arithmetic",
                normalisation="ami",
            )
        except Exception as e:
            logger.exception(f"[{data_name}] AMI evaluation failed at epoch {epoch}: {e}")
            epoch_ami = float("-inf")

        epoch_amis.append(epoch_ami)
        logger.info(f'[{data_name}] Epoch AMI: {epoch_ami}')

        if epoch_ami > best_ami:
            best_ami = epoch_ami
            best_epoch = epoch
            best_state_dict = copy.deepcopy(tgn.state_dict())
            best_model_path = f"saved_models/best_tgn_mask_by_ami_{data_name}.pth"
            torch.save(best_state_dict, best_model_path)
            logger.info(f"[{data_name}] New best model saved at epoch {epoch}, AMI={epoch_ami:.6f}")

    return {
        "data_name": data_name,
        "best_ami": best_ami,
        "best_epoch": best_epoch,
        "epoch_amis": epoch_amis,
        "train_losses": train_losses,
    }


syn_files = sorted(glob.glob(os.path.join("syn_data", "*.csv")))
all_results = []

for syn_path in syn_files:
    data_name = os.path.splitext(os.path.basename(syn_path))[0]
    print(f"\n===== Training on {data_name} =====")
    try:
        result = train_one_dataset(
            data_name=data_name,
            preprocess_dir="preprocess",
            pretrain_dir="pretrain",
        )
        all_results.append(result)
        print(f"[{data_name}] best_ami = {result['best_ami']}")
    except Exception as e:
        logger.exception(f"Failed on dataset {data_name}: {e}")
        all_results.append({
            "data_name": data_name,
            "best_ami": float("-inf"),
            "best_epoch": -1,
            "epoch_amis": [],
            "train_losses": [],
            "error": str(e),
        })

results_df = pd.DataFrame([
    {
        "data_name": r["data_name"],
        "best_ami": r["best_ami"],
        "best_epoch": r["best_epoch"],
        "error": r.get("error", ""),
    }
    for r in all_results
])
results_df.to_csv("saved_models/all_dataset_best_ami.csv", index=False)
print(results_df)