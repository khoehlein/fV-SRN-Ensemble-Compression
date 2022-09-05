import os
import struct
import subprocess
import tempfile
import time
from enum import Enum
from typing import List, Any, Tuple, Optional

import numpy as np
import zfpy


STRUCT_DTYPE_FLAGS = {
    np.dtype(float): 'f',
    np.dtype(np.float32): 'f',
    np.dtype(np.float64): 'd',
    np.dtype(int): 'i',
    np.dtype(np.int32): 'i',
    np.dtype(np.int64): 'l',
}


class ICompressor(object):

    def encode(self, x: np.ndarray) -> Any:
        raise NotImplementedError()

    def decode(self, code: Any, shape: Tuple[int, ...], dtype: np.dtype):
        raise NotImplementedError()


class _CLICompressor(ICompressor):

    DATA_FILE_NAME = 'data.dat'
    COMPRESSED_FILE_NAME = 'array.dat.compressed'

    def __init__(self, verbose: bool):
        self.verbose = verbose

    def _get_encode_command(self, compressed_file_name, data_file_name, shape: Tuple[int, ...], dtype: np.dtype):
        raise NotImplementedError()

    def _get_decode_command(self, compressed_file_name, data_file_name, shape: Tuple[int, ...], dtype: np.dtype):
        raise NotImplementedError()

    def _get_tmp_dir(self):
        # if platform.system() == 'Linux':
        #      return os.path.join('/run', 'user', f'{os.getuid()}')
        return None

    def encode(self, x: np.ndarray):
        data = x.ravel().tolist()
        shape = x.shape
        dtype = x.dtype
        assert len(shape) <= 4, '[ERROR] Data shape must be of dimension 4 or less.'
        assert len(data) == np.prod(shape), f'[ERROR] Number of data entries ({len(data)}) must agree with field entries in array of given shape {shape}.'
        t_start = time.time()
        with tempfile.TemporaryDirectory(dir=self._get_tmp_dir()) as temp_dir:
            if self.verbose:
                print(f'[INFO] Created temporary directory: {temp_dir}')
            data_file_name = self._write_data_file(temp_dir, data, dtype)
            compressed_file_name, _ = self._encode_data_file(data_file_name, shape, dtype)
            code = self._read_compressed_file(compressed_file_name)
        t_end = time.time()
        if self.verbose:
            print(f'[INFO] Total time was {t_end - t_start} s.')
            print(f'[INFO] Code length: {len(code)} bytes')
        return code

    def _write_data_file(self, temp_dir, data, dtype: np.dtype):
        s = struct.pack(STRUCT_DTYPE_FLAGS[dtype] * len(data), *data)
        data_file_name = os.path.join(temp_dir, self.DATA_FILE_NAME)
        with open(data_file_name, 'wb') as f:
            f.write(s)
        return data_file_name

    def _encode_data_file(self, data_file_name, shape, dtype: np.dtype):
        temp_dir = os.path.split(data_file_name)[0]
        compressed_file_name = os.path.join(temp_dir, self.COMPRESSED_FILE_NAME)
        command = self._get_encode_command(compressed_file_name, data_file_name, shape, dtype)
        if self.verbose:
            print('[INFO] Calling compressor:', *command)
            process = subprocess.Popen(command)
        else:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.communicate()
        return compressed_file_name, output

    def _read_compressed_file(self, compressed_file_name):
        with open(compressed_file_name, 'rb') as f:
            code = f.read()
        return code

    def decode(self, code: List[Any], shape: Tuple[int, ...], dtype: np.dtype):
        assert len(shape) <= 4, '[ERROR] Data shape must be of dimension 4 or less.'
        t_start = time.time()
        with tempfile.TemporaryDirectory(dir=self._get_tmp_dir()) as temp_dir:
            if self.verbose:
                print(f'[INFO] Created temporary directory: {temp_dir}')
            compressed_file_name = self._write_compressed_file(temp_dir, code)
            data_file_name, _ = self._decode_compressed_file(compressed_file_name, shape, dtype)
            data = self._read_data_file(data_file_name, shape, dtype)
        t_end = time.time()
        if self.verbose:
            print(f'[INFO] Total time was {t_end - t_start} s.')
        return data

    def _write_compressed_file(self, temp_dir, code):
        compressed_file_name = os.path.join(temp_dir, self.COMPRESSED_FILE_NAME)
        with open(compressed_file_name, 'wb') as f:
            f.write(code)
        return compressed_file_name

    def _decode_compressed_file(self, compressed_file_name, shape, dtype: np.dtype):
        temp_dir = os.path.split(compressed_file_name)[0]
        data_file_name = os.path.join(temp_dir, self.DATA_FILE_NAME)
        command = self._get_decode_command(compressed_file_name, data_file_name, shape, dtype)
        if self.verbose:
            print('[INFO] Calling compressor:', *command)
            process = subprocess.Popen(command)
        else:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.communicate()
        return data_file_name, output

    def _read_data_file(self, data_file_name, shape, dtype: np.dtype):
        num_values = int(np.prod(shape))
        with open(data_file_name, 'rb') as f:
            s = f.read()
        data = struct.unpack(STRUCT_DTYPE_FLAGS[dtype] * num_values, s)
        return data


class SZ3(_CLICompressor):

    EXE_PATH = 'sz3' #'/home/hoehlein/software/SZ3/bin/sz3'

    class CompressionMode(Enum):
        ABS = 'ABS'
        REL = 'REL'
        PSNR = 'PSNR'
        NORM = 'NORM'
        ABS_AND_REL = 'ABS_AND_REL'
        ABS_OR_REL = 'ABS_OR_REL'

    DTYPE_FLAGS = {
        np.dtype(float): '-f',
        np.dtype(np.float32): '-f',
        np.dtype(np.float64): '-d',
        np.dtype(int): '-I 32',
        np.dtype(np.int32): '-I 32',
        np.dtype(np.int64): '-I 64',
    }

    def __init__(self, mode: 'SZ3.CompressionMode', threshold: float = None, verbose: bool = False):
        super(SZ3, self).__init__(verbose)
        self.mode = mode
        self.threshold = threshold
        self.verbose = verbose

    def _get_sz3_command(self, file_specs, shape, dtype: np.dtype):
        dim = len(shape)
        prefix = [SZ3.EXE_PATH, *SZ3.DTYPE_FLAGS[dtype].split(' ')]
        suffix = [f'-{dim}', *[f'{s}' for s in reversed(shape)], '-M', self.mode.value, f'{self.threshold}']
        command = [*prefix, *file_specs, *suffix]
        return command

    def _get_encode_command(self, compressed_file_name, data_file_name, shape: Tuple[int, ...], dtype: np.dtype):
        file_specs = ['-i', data_file_name, '-z', compressed_file_name]
        return self._get_sz3_command(file_specs, shape, dtype)

    def _get_decode_command(self, compressed_file_name, data_file_name, shape: Tuple[int, ...], dtype: np.dtype):
        file_specs = ['-z', compressed_file_name, '-o', data_file_name]
        return self._get_sz3_command(file_specs, shape, dtype)


class TTHRESH(_CLICompressor):

    EXE_PATH = '/home/hoehlein/PycharmProjects/third-party/tthresh/build/tthresh' #'/home/hoehlein/software/TTHRESH/bin/tthresh'

    class CompressionMode(Enum):
        RMSE = '-r'
        REL = '-e'
        PSNR = '-p'

    DTYPE_FLAGS = {
        np.dtype(float): 'float',
        np.dtype(np.float32): 'float',
        np.dtype(np.float64): 'double',
        np.dtype(int): 'int',
        np.dtype(np.int32): 'int',
    }

    def __init__(self, mode: 'TTHRESH.CompressionMode', threshold: float = None, verbose: bool = False):
        super(TTHRESH, self).__init__(verbose)
        self.mode = mode
        self.threshold = threshold
        self.verbose = verbose

    def _get_encode_command(self, compressed_file_name, data_file_name, shape: Tuple[int, ...], dtype: np.dtype):
        prefix = [self.EXE_PATH, '-t', TTHRESH.DTYPE_FLAGS[dtype]]
        file_specs = ['-i', data_file_name, '-c', compressed_file_name]
        suffix = ['-s', *[f'{s}' for s in reversed(shape)], self.mode.value, '{:f}'.format(self.threshold)]
        command = [*prefix, *file_specs, *suffix]
        if self.verbose:
            command.append('-v')
        return command

    def _get_decode_command(self, compressed_file_name, data_file_name, shape: Tuple[int, ...], dtype: np.dtype):
        command = [self.EXE_PATH, '-c', compressed_file_name, '-o', data_file_name]
        if self.verbose:
            command.append('-v')
        return command


class ZFP(ICompressor):

    class CompressionMode(Enum):
        ABS = 'tolerance'
        REL = 'precision'
        MEM = 'rate'
        REV = 'reversible'

    def __init__(self, mode: 'ZFP.CompressionMode', threshold: Optional[float] = None):
        self.mode = mode
        self.threshold = threshold

    def encode(self, x: np.ndarray) -> Any:
        if self.mode != ZFP.CompressionMode.REV:
            kwargs = {self.mode.value: self.threshold, 'write_header': False}
        else:
            kwargs = {'write_header': False}
        code = zfpy.compress_numpy(x, **kwargs)
        return code

    def decode(self, code: Any, shape: Tuple[int, ...], dtype: np.dtype):
        if self.mode != ZFP.CompressionMode.REV:
            kwargs = {self.mode.value: self.threshold}
        else:
            kwargs = {}
        data = zfpy._decompress(code, zfpy.dtype_to_ztype(dtype), shape, **kwargs)
        return data