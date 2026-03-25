import numpy as np

### Utility function and class
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1
        
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round

class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

# New obj
class RandNodeTimeSampler(object):
    def __init__(self, src_list, dst_list, ts_list, unique=True):
        src_arr = np.asarray(src_list)
        dst_arr = np.asarray(dst_list)
        ts_arr = np.asarray(ts_list)

        src_pairs = np.column_stack((src_arr, ts_arr))
        dst_pairs = np.column_stack((dst_arr, ts_arr))
        node_time_pairs = np.concatenate([src_pairs, dst_pairs], axis=0)

        if unique:
            node_time_pairs = np.unique(node_time_pairs, axis=0)

        self.node_time_pairs = node_time_pairs
        self.node_dtype = src_arr.dtype
        self.ts_dtype = ts_arr.dtype

    def sample(self, size):
        sampled_idx = np.random.randint(0, len(self.node_time_pairs), size)
        sampled_pairs = self.node_time_pairs[sampled_idx]
        sampled_nodes = sampled_pairs[:, 0].astype(self.node_dtype, copy=False)
        sampled_times = sampled_pairs[:, 1].astype(self.ts_dtype, copy=False)
        return sampled_nodes, sampled_times