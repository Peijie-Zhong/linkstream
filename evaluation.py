from __future__ import annotations
from typing import Dict, Hashable, Tuple, Any, Optional, Literal
import pandas as pd
from typing import Dict, Hashable, Iterable, Tuple, Any, Optional
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score
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

    for s, d, t, cs, cd in df[[source_col, destination_col, timestamp_col, source_commu_col, destination_commu_col]].itertuples(index=False, name=None):
        _assign((s, t), cs)
        _assign((d, t), cd)

    return part


def dynamic_mi(
    gt: Dict[Key, Any],
    pred: Dict[Key, Any],
    keys: Optional[Iterable[Key]] = None,
    average_method: str = "arithmetic",
    normalisation: str = "ami" or "nmi"
) -> float:
    if keys is None:
        keys = set(gt.keys()) | set(pred.keys())

    keys = list(keys)

    y_true = []
    y_pred = []
    for k in keys:
        if k not in gt or k not in pred:
            raise KeyError(f"node {k} is missing in either gt or pred")
        y_true.append(gt[k])
        y_pred.append(pred[k])

    if len(y_true) == 0:
        raise ValueError("no samples left after filtering missing nodes")

    if normalisation == "ami":
        return float(adjusted_mutual_info_score(y_true, y_pred, average_method=average_method))
    elif normalisation == "nmi":
        return float(normalized_mutual_info_score(y_true, y_pred, average_method=average_method)) 
    else:
        return ("AttributeError")



def dynamic_ari(
    gt: Dict[Key, Any],
    pred: Dict[Key, Any],
    keys: Optional[Iterable[Key]] = None,
) -> float:
    """
    Compute Adjusted Rand Index (ARI) between two dynamic partitions.

    Parameters
    ----------
    gt : dict
        Ground-truth partition, mapping (node, time) -> label
    pred : dict
        Predicted partition, mapping (node, time) -> label
    keys : iterable, optional
        Subset of (node, time) keys to evaluate on.
        If None, evaluate on the union of keys from gt and pred.

    Returns
    -------
    float
        ARI score
    """
    if keys is None:
        keys = set(gt.keys()) | set(pred.keys())

    keys = list(keys)

    y_true = []
    y_pred = []
    for k in keys:
        if k not in gt or k not in pred:
            raise KeyError(f"node {k} is missing in either gt or pred")
        y_true.append(gt[k])
        y_pred.append(pred[k])

    if len(y_true) == 0:
        raise ValueError("no samples left after filtering missing nodes")

    return float(adjusted_rand_score(y_true, y_pred))
