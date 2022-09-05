import numpy as np

from .compressors import ICompressor, SZ3


class CompressedArray(object):

    def __init__(self, code, shape, dtype: np.dtype, compressor: ICompressor):
        self.code = code
        self.shape = shape
        self.dtype = dtype
        self.compressor = compressor

    @classmethod
    def from_numpy(cls, x: np.ndarray, compressor: ICompressor):
        shape = x.shape
        dtype = x.dtype
        code = compressor.encode(x)
        return cls(code, shape, dtype, compressor)

    def restore_numpy(self):
        data = self.compressor.decode(self.code, self.shape, self.dtype)
        data = np.array(data).reshape(self.shape)
        return data

    def code_size(self):
        return len(self.code)

    def data_size(self):
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def compression_ratio(self):
        return self.data_size() / self.code_size()


def _test():
    import matplotlib.pyplot as plt

    data = 0.01 * np.random.randn(10, 10, 20) + np.arange(20)[None, None, :]

    for acc in [1., .1, .01, .001, .0001, .00001]:
        compressor = SZ3(SZ3.CompressionMode.NORM, acc, verbose=True)
        compressed = CompressedArray.from_numpy(data, compressor)
        reconstruction = compressed.restore_numpy()
        deviation = np.abs(reconstruction - data)
        rse = np.sqrt(np.sum(deviation ** 2))

        plt.figure()
        plt.hist(deviation.ravel(), bins=50)
        plt.gca().set(title=f'threshold = {acc}, rse = {rse}, compression = {compressed.compression_ratio()}')
        plt.show()
        plt.close()


if __name__== '__main__':
    _test()