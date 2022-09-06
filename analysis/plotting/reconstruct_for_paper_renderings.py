import os

import numpy as np
import torch
import pyrenderer

from analysis.statistics import CompressionStatistics
from data.datasets.univariate import WorldSpaceDensityEvaluator
from inference.compression import CompressedArray, TTHRESH, SZ3, ZFP


def _export_volume_data(data, path):
    print(data.shape)
    vol = pyrenderer.Volume()
    vol.worldX = 10.
    vol.worldY = 10.
    vol.worldZ = 1.
    vol.add_feature_from_tensor('tk', data)
    vol.save(path)


def _evaluate_network(model_path, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    checkpoint = torch.load(model_path, map_location='cpu')
    network = checkpoint['model']
    resolution = (250, 352, 12)
    positions = np.meshgrid(*[(np.arange(r) + 0.5) / r for r in resolution], indexing='ij')
    positions = np.stack([p.astype(np.float32).ravel() for p in positions], axis=-1)
    positions = torch.from_numpy(positions)
    print(f'[INFO] Compression rate: {CompressionStatistics(network).compression_rate()}')

    with torch.no_grad():
        for i in range(3):
            evaluator = WorldSpaceDensityEvaluator(network, 0, 0, i)
            predictions = evaluator.evaluate(positions)
            print(predictions.shape)
            out = predictions.view(1, *resolution)
            path = os.path.join(target_folder, 'm{:02d}.cvol'.format(i + 1))
            _export_volume_data(out, path)


def evaluate_multi_grid_model():
    model_path = '/home/hoehlein/PycharmProjects/results/fvsrn/paper/ensemble/multi_grid/resampling/12-22-16_32-32-32-32_1-65/results/model/run00004/model_epoch_50.pth'
    # old path: '/home/hoehlein/PycharmProjects/results/fvsrn/paper/ensemble/multi_grid/num_channels/6-44-31_64_1-65_fast/results/model/run00005/model_epoch_50.pth'
    target_folder = '/home/hoehlein/Desktop/rendering_data/quality_250/tk/multi_grid'
    _evaluate_network(model_path, target_folder)


def evaluate_multi_core_model():
    model_path = '/home/hoehlein/PycharmProjects/results/fvsrn/paper/ensemble/multi_core/resampling/12-22-16_32-32-32-32_1-65/results/model/run00004/model_epoch_50.pth'
    # old path: '/home/hoehlein/PycharmProjects/results/fvsrn/paper/ensemble/multi_core/num_channels/6-44-31_64_1-65_fast/results/model/run00012/model_epoch_50.pth'
    target_folder = '/home/hoehlein/Desktop/rendering_data/quality_250/tk/multi_core'
    _evaluate_network(model_path, target_folder)


def evaluate_compressor(compressor, name):
    source_folder = '/home/hoehlein/Desktop/rendering_data/quality/tk/ground_truth'
    target_folder = f'/home/hoehlein/Desktop/rendering_data/quality_250/tk/{name}'
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    out = []
    for f in os.listdir(source_folder):
        vol = pyrenderer.Volume(os.path.join(source_folder, f))
        data = vol.get_feature(0).get_level(0).to_tensor()[0].data.cpu().numpy()
        compressed = CompressedArray.from_numpy(data, compressor)
        print(f'[INFO] Compression ratio: {compressed.compression_ratio()}')
        out.append(compressed.compression_ratio())
        restored = torch.from_numpy(compressed.restore_numpy())[None, ...].to(torch.float32)
        path = os.path.join(target_folder, f)
        _export_volume_data(restored, path)
    print('[INFO] Compression ratio (average):', 1. / np.mean(1. / np.array(out)))


def evaluate_compressor_new(compressor, name):
    source_folder = '/home/hoehlein/Desktop/rendering_data/quality/tk/ground_truth'
    target_folder = f'/home/hoehlein/Desktop/rendering_data/quality_250/tk/{name}'
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    out = []
    data = []
    for f in os.listdir(source_folder):
        vol = pyrenderer.Volume(os.path.join(source_folder, f))
        data.append(vol.get_feature(0).get_level(0).to_tensor()[0].data.cpu().numpy())
    data = torch.stack(data, )
    compressed = CompressedArray.from_numpy(data, compressor)
    print(f'[INFO] Compression ratio: {compressed.compression_ratio()}')
    out.append(compressed.compression_ratio())
    restored = torch.from_numpy(compressed.restore_numpy())[None, ...].to(torch.float32)
    path = os.path.join(target_folder, f)
    _export_volume_data(restored, path)
    print(1. / np.mean(1. / np.array(out)))


def evaluate_tthresh():
    # first submission: compressor = TTHRESH(TTHRESH.CompressionMode.RMSE, 6.5e-3, verbose=True)
    compressor = TTHRESH(TTHRESH.CompressionMode.RMSE, 6.91e-3, verbose=True)
    evaluate_compressor(compressor, 'tthresh')


def evaluate_sz3():
    # first submission: compressor = SZ3(SZ3.CompressionMode.ABS, 6.e-2, verbose=True)
    compressor = SZ3(SZ3.CompressionMode.ABS, 6.2e-2, verbose=True)
    evaluate_compressor(compressor, 'sz3')


def evaluate_zfp():
    compressor = ZFP(ZFP.CompressionMode.ABS, 5.)
    evaluate_compressor(compressor, 'zfp')


if __name__ =='__main__':
    evaluate_sz3() # 248.81x
    # evaluate_zfp()
    evaluate_tthresh() # 253.25x
    evaluate_multi_core_model() # 251.88x
    evaluate_multi_grid_model() # 248.00x
