import atexit
import re

import collections
import torch
import sys
import os
import numpy as np
from typing import Tuple, Union
from warnings import warn

from torch._six import string_classes

_PYRENDERER_BASE_PATH = '../external/pyrenderer/bin'

try:
    import pyrenderer
except ModuleNotFoundError:
    __newpath = os.path.abspath(os.path.join(os.path.split(__file__)[0], _PYRENDERER_BASE_PATH))
    sys.path.append(__newpath)
    print("Search pyrenderer in '%s'" % __newpath)
    import pyrenderer
print("pyrenderer native library loaded")


@atexit.register
def __cleanup_renderer():
    pyrenderer.cleanup()


def make_float3(vector):
    return pyrenderer.float3(vector[0], vector[1], vector[2])


def make_float4(vector):
    return pyrenderer.float4(vector[0], vector[1], vector[2], vector[3])


def make_double3(vector):
    return pyrenderer.double3(vector[0], vector[1], vector[2])


def make_double4(vector):
    return pyrenderer.double4(vector[0], vector[1], vector[2], vector[3])


def copy_double3(v: pyrenderer.double3):
    return pyrenderer.double3(v.x, v.y, v.z)


def copy_double4(v: pyrenderer.double4):
    return pyrenderer.double3(v.x, v.y, v.z, v.w)


def cvector_to_numpy(vector: Union[pyrenderer.float3, pyrenderer.float4, pyrenderer.double3, pyrenderer.double4]):
    if isinstance(vector, pyrenderer.float3):
        return np.array([vector.x, vector.y, vector.z], dtype=np.float32)
    elif isinstance(vector, pyrenderer.float4):
        return np.array([vector.x, vector.y, vector.z, vector.w], dtype=np.float32)
    if isinstance(vector, pyrenderer.double3):
        return np.array([vector.x, vector.y, vector.z], dtype=np.float64)
    elif isinstance(vector, pyrenderer.double4):
        return np.array([vector.x, vector.y, vector.z, vector.w], dtype=np.float64)
    else:
        raise ValueError("unsupported type, real3 or real4 expected but got", type(vector))


def inverseSigmoid(y):
    """
  inverse of y=torch.sigmoid(y)
  :param y:
  :return: x
  """
    return torch.log(-y / (y - 1))


class InverseSigmoid(torch.nn.Module):
    def forward(self, y):
        return inverseSigmoid(y)


def inverseSoftplus(y, beta=1, threshold=20):
    """
  inverse of y=torch.nn.functional.softplus(x, beta, threshold)
  :param y: the output of the softplus
  :param beta: the smoothness of the step
  :param threshold: the threshold after which a linear function is used
  :return: the input
  """
    return torch.where(y * beta > threshold, y, torch.log(torch.exp(beta * y) - 1) / beta)


class InverseSoftplus(torch.nn.Module):
    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self._beta = beta
        self._threshold = threshold

    def forward(self, y):
        return inverseSoftplus(y, self._beta, self._threshold)


def implies(x, y):
    """
  Returns "x implies y" / "x => y"
  """
    return not (x) or y


def toCHW(bhwc: torch.Tensor):
    """
  Converts a tensor in BxHxWxC to BxCxHxW
  """
    return bhwc.movedim((0, 1, 2, 3), (0, 2, 3, 1))


def toHWC(bchw: torch.Tensor):
    """
  Converts a tensor in BxCxHxW to BxHxWxC
  """
    return bchw.movedim((0, 2, 3, 1), (0, 1, 2, 3))


def toDevice(tensor_list, device: torch.device):
    """
  Converts a tensor or a list/tuple of theireof to the given device
  :param tensor_list:
  :return:
  """
    if isinstance(tensor_list, torch.Tensor):
        return tensor_list.to(device=device)
    elif isinstance(tensor_list, list):
        return [toDevice(d, device) for d in tensor_list]
    elif isinstance(tensor_list, tuple):
        return tuple([toDevice(d, device) for d in tensor_list])
    else:
        raise ValueError("Unknown collection class: " + str(type(tensor_list)))


def fibonacci_sphere(N: int, *, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """
  Generates points on a sphere using the Fibonacci spiral
  :param N: the number of points
  :return: a tuple (pitch/latitude, yaw/longitude)
  """
    # Source: https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html
    gr = (np.sqrt(5.0) + 1.0) / 2.0  # golden ratio = 1.618...
    ga = (2 - gr) * (2 * np.pi)  # golden angle = 2.399...
    i = np.arange(1, N + 1, dtype=dtype)
    lat = np.arcsin(-1 + 2 * i / (N + 1))
    lon = np.remainder(ga * i, 2 * np.pi)
    # lon = np.arcsin(np.sin(ga*i))
    return lat, lon


def safe_normalize(a: torch.Tensor, axis=-1, epsilon: float = 1e-4):
    l2 = torch.norm(a, dim=axis, keepdim=True)
    l2 = torch.clip(l2, min=epsilon)
    return a / l2


deprecated_names = ["make_real3", "make_real4", "renderer_dtype_np", "renderer_dtype_torch"]


def __getattr__(name):
    if name in deprecated_names:
        warn(f"{name} is deprecated", DeprecationWarning)
        return globals()[f"_deprecated_{name}"]
    raise AttributeError(f"module {__name__} has no attribute {name}")


def humanbytes(num, suffix='B', multiplier=1024.0):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < multiplier:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= multiplier
    return "%.1f%s%s" % (num, 'Yi', suffix)


def next_multiple(a, b):
    m = a % b
    if m == 0: return a
    return a + b - m


def div_up(num, denum):
    """
  Performs integer division num/denum, but rounds the result UP.
  Source: https://stackoverflow.com/a/35125872/1786598
  :param num:
  :param denum:
  :return:
  """
    return -(-num // denum)


def parse_range_string(range_string: str):
    return [i for i in range(*map(int, range_string.split(':')))]


def parse_slice_string(slice_string: str):
    def int_or_none(x):
        try:
            out = int(x)
        except ValueError:
            out = None
        return out

    out = tuple(map(int_or_none, slice_string.split(':')))
    return out


def is_integer_index(idx):
    try:
        int_idx = int(idx)
    except Exception():
        return False
    else:
        return int_idx == idx


def is_valid_index_position(pos, idx):
    l = len(idx)
    if pos < -l:
        return False
    if pos >= l:
        return False
    return True


def get_normalized_positions(num_nodes, clamp=False):
    num_segments = 2 * (num_nodes - 1)
    positions = torch.arange(1, num_segments + 2, 2) / num_segments
    if clamp:
        positions[-1] = 1.
    return positions


_np_str_obj_array_pattern = re.compile(r'[SaUO]')
_cat_collate_err_msg_format = "_cat_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}"


def cat_collate(batch):
    r"""
    Puts each data field into a tensor with outer dimension batch size.
    Uses concatenation instead of stacking -> elements are already partially batched
    """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if _np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(_cat_collate_err_msg_format.format(elem.dtype))

            return cat_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: cat_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(cat_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [cat_collate(samples) for samples in transposed]

    raise TypeError(_cat_collate_err_msg_format.format(elem_type))
