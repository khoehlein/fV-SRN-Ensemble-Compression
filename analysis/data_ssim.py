import numpy as np
from scipy import signal


class GaussianFilter2d(object):

    def __init__(self, sigma, k, dtype=np.float64):
        self._sigma = sigma
        self._k = k
        self._dtype = dtype
        self.kernel = self._get_kernel()

    def __call__(self, data: np.ndarray):
        shape = [1 for _ in data.shape[:-2]] + list(self.kernel.shape)
        return signal.correlate(data, np.reshape(self.kernel, shape), mode='valid', method='auto')

    def _get_kernel(self):
        x = (np.arange(self._k) - (self._k - 1) / 2.) / self._sigma
        k1d = np.exp(- x ** 2 / 2.)
        k2d = k1d[:, None] * k1d[None, :]
        k2d = k2d.astype(self._dtype)
        return k2d / np.sum(k2d)


class DSSIM2d(object):

    def __init__(self, k=11, sigma=1.5, c1=1.e-8, c2=1.e-8, num_bins=256):
        self.filter = GaussianFilter2d(sigma, k)
        self._c1 = c1
        self._c2 = c2
        self._num_bins = num_bins

    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        x1, x2 = self._normalize_data(x1, x2)
        x1, x2 = self._discretize_data(x1, x2)
        return self._compute_ssim(x1, x2)

    def _compute_ssim(self, x1, x2):
        x1 = x1.astype(np.float64)
        x2 = x2.astype(np.float64)
        mu_x1 = self.filter(x1)
        mu_x2 = self.filter(x2)
        mu_x1sq = self.filter(x1 ** 2)
        mu_x2sq = self.filter(x2 ** 2)
        mu_x1x2 = self.filter(x1 * x2)

        mu_x1_sq = mu_x1 ** 2
        mu_x2_sq = mu_x2 ** 2
        mu_x1_mu_x2 = mu_x1 * mu_x2

        var_x1 = mu_x1sq - mu_x1_sq
        var_x2 = mu_x2sq - mu_x2_sq
        cov_x1x2 = mu_x1x2 - mu_x1_mu_x2

        t = 2. * mu_x1_mu_x2 + self._c1
        b = mu_x1_sq + mu_x2_sq + self._c1
        ssim_1 = t / b

        t = 2. * cov_x1x2 + self._c2
        b = var_x1 + var_x2 + self._c2
        ssim_2 = t / b
        ssim_mat = ssim_1 * ssim_2
        # cropping not required due to scipy.signal.convolve with mode valid
        return np.nanmean(ssim_mat, axis=(-1, -2))

    def _normalize_data(self, *data: np.ndarray):
        min_val = min([np.nanmin(x) for x in data])
        max_val = max([np.nanmax(x) for x in data])
        r = max_val - min_val

        def _normalize(x: np.ndarray):
            if r == 0.:
                if max_val == 0.:
                    return x
                else:
                    return x / max_val
            else:
                return (x - min_val) /max_val

        return tuple(_normalize(x) for x in data)

    def _discretize_data(self, *data: np.ndarray):
        return tuple(np.round(x * (self._num_bins - 1)) / (self._num_bins - 1) for x in data)


def _test():
    a = np.random.randn(10, 10, 50, 50)
    b = np.random.randn(10, 10, 50, 50)
    dssim = DSSIM2d()
    out = dssim(a, - a + 1.e-6 * b)
    print(out)


if __name__ == '__main__':
    _test()
