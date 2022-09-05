import argparse
import torch
import os
import numpy as np

import common.utils
import pyrenderer

from inference.model.pyrenderer import PyrendererSRN


def export(checkpoint_file: str, compiled_file_prefix: str,
           world_size_x:float=10, world_size_y:float=10, world_size_z:float=1):
    state = torch.load(checkpoint_file)
    print("state keys:", state.keys())
    model = state['model']
    print(model)
    print(model.__dir__())
    assert isinstance(model, PyrendererSRN)
    if model.uses_time():
        raise ValueError("Time dependency not supported")
    if model.output_channels() != 1:
        raise ValueError("Only a single output channel supported, not %d"%model.output_channels())
    num_members = model.num_members()
    print("Num members:", num_members)

    resolution = (250, 352, 12)
    device = torch.device("cuda")
    positions = np.meshgrid(*[(np.arange(r) + 0.5) / r for r in resolution], indexing='ij')
    positions = np.stack([p.astype(np.float32).ravel() for p in positions], axis=-1)
    positions = torch.from_numpy(positions).to(device=device)
    volume_network = pyrenderer.VolumeInterpolationNetwork()

    #grid_encoding = pyrenderer.SceneNetwork.LatentGrid.Float
    grid_encoding = pyrenderer.SceneNetwork.LatentGrid.ByteLinear
    for m in range(num_members):
        net = model.export_to_pyrenderer(grid_encoding, ensemble=m)
        filename = compiled_file_prefix + "-ensemble%03d.volnet"%m
        net.box_min = pyrenderer.float3(-world_size_x/2, -world_size_y/2, -world_size_z/2)
        net.box_size = pyrenderer.float3(world_size_x, world_size_y, world_size_z)
        net.save(filename)
        print(f"Saved ensemble {m} to {filename}")

        filename = compiled_file_prefix + "-ensemble%03d.cvol" % m
        volume_network.set_network(net)
        predictions = volume_network.evaluate(positions)
        out = predictions.view(1, *resolution)
        vol = pyrenderer.Volume()
        vol.worldX = 10.
        vol.worldY = 10.
        vol.worldZ = 1.
        vol.add_feature_from_tensor('tk', out.cpu())
        vol.save(filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-directory', type=str, required=True)
    parser.add_argument('--checkpoint-file-name', type=str, required=True)
    args = vars(parser.parse_args())

    checkpoint_base_dir = os.path.join(args['experiment_directory'], 'results', 'model')
    runs = os.listdir(checkpoint_base_dir)
    volnet_base_dir = os.path.join(args['experiment_directory'], 'results', 'volnet')

    for run in runs:
        print(run)
        volnet_dir = os.path.join(volnet_base_dir, run)
        if not os.path.isdir(volnet_dir):
            os.makedirs(volnet_dir)
        checkpoint_path = os.path.join(checkpoint_base_dir, run, args['checkpoint_file_name'])
        output_file_prefix = os.path.join(volnet_dir, os.path.splitext(args['checkpoint_file_name'])[0])
        export(checkpoint_path, output_file_prefix)


if __name__ == '__main__':
    main()
