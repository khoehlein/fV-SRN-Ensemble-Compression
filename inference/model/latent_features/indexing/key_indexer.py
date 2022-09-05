import torch
from torch import Tensor, nn


class KeyIndexer(nn.Module):

    def query(self, keys: Tensor):
        sorted_keys, order = torch.sort(keys)
        unique_keys, counts = self._compute_decomposition(sorted_keys)
        segments = torch.split(order, counts.tolist())
        return unique_keys, segments

    @staticmethod
    def _compute_decomposition(sorted_keys: Tensor):
        steps = torch.arange(1, len(sorted_keys))[torch.not_equal(torch.diff(sorted_keys), 0)]
        unique_keys = torch.full((len(steps) + 1,), sorted_keys[0], dtype=sorted_keys.dtype, device=sorted_keys.device)
        unique_keys[1:] = sorted_keys[steps]
        all_steps = torch.zeros(len(steps) + 2, device=steps.device, dtype=steps.dtype)
        all_steps[1:-1] = steps
        all_steps[-1] = len(sorted_keys)
        counts = torch.diff(all_steps).to(dtype=torch.long)
        return unique_keys, counts


def _test_indexer():
    keys = torch.randint(20, size=(10000,))
    indexer = KeyIndexer()
    out = indexer.query(keys)
    print('Finished')


if __name__ == '__main__':
    _test_indexer()
