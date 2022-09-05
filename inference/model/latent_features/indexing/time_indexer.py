from typing import List, Any
import torch
from torch import nn, Tensor


class TimeIndexer(nn.Module):

    def __init__(self, key_times: List[Any], device=None):
        super(TimeIndexer, self).__init__()
        key_times, _ = torch.sort(torch.tensor(key_times, device=device))
        self.register_buffer('key_times', key_times)

    def query(self, time: Tensor):
        indices = torch.searchsorted(self.key_times, time)
        upper = torch.clamp(indices, 0, self.num_key_times() - 1)
        lower = torch.where(
            torch.lt(time, self.max_time()),
            torch.clamp(indices - 1, 0, self.num_key_times() - 1),
            torch.full_like(indices, self.num_key_times() - 1)
        )
        fraction = torch.zeros_like(time)
        bounds_differ = torch.not_equal(lower, upper)
        time_lower = self.key_times[lower[bounds_differ]]
        time_upper = self.key_times[upper[bounds_differ]]
        fraction[bounds_differ] = (time[bounds_differ] - time_lower) / (time_upper - time_lower)
        return lower, upper, fraction

    def num_key_times(self):
        return len(self.key_times)

    def min_time(self):
        return self.key_times[0].item()

    def max_time(self):
        return self.key_times[-1].item()


def _test_indexer():
    indexer = TimeIndexer(torch.tensor([0., 1.]))
    times = torch.tensor([-1., 0., 0.5, 1., 2.])
    lower, upper = indexer.get_bounds(times)
    print('Finished')


if __name__ == '__main__':
    _test_indexer()