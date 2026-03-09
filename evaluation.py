from typing import Dict, Hashable, Iterable, Tuple, Any, Optional, Literal
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

Node = Hashable
Time = Hashable
Key = Tuple[Node, Time]


def dynamic_mi(
    gt: Dict[Key, Any],
    pred: Dict[Key, Any],
    keys: Optional[Iterable[Key]] = None,
    average_method: str = "arithmetic",
    normalisation: Literal["ami", "nmi"] = "ami",
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
        raise AttributeError(f"Unknown normalisation: {normalisation}")