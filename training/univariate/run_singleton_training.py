import argparse
import os.path
import time
from collections import defaultdict
from contextlib import ExitStack
from typing import Dict, Any

import numpy as np
import torch
import tqdm

from data.datasets import DatasetType
from data.datasets.resampling.resampler import DatasetResampler
from data.datasets.sampling import PositionSampler
from data.datasets.univariate import VolumeDataStorage, WorldSpaceDensityData, WorldSpaceVisualizationData
from inference.losses.lossnet import LossFactory
from inference.model.pyrenderer import PyrendererOutputParameterization, PyrendererSRN
from inference.volume import VolumeEvaluator, RenderTool
from training.evaluation import EvaluateWorld, EvaluateScreen
from training.optimizer import Optimizer
from training.in_out.storage_manager import StorageManager
from training.profiling import build_profiler
from training.visualization import Visualizer

torch.set_num_threads(12)
device = torch.device('cuda:0')

from common import utils

# align model output with dataset output
PyrendererOutputParameterization.set_output_mode(
    VolumeEvaluator.output_mode()
)


def build_parser():
    parser = argparse.ArgumentParser()
    StorageManager.init_parser(parser, os.path.split(__file__)[0])
    VolumeDataStorage.init_parser(parser)
    PositionSampler.init_parser(parser)
    WorldSpaceDensityData.init_parser(parser)
    WorldSpaceVisualizationData.init_parser(parser)
    DatasetResampler.init_parser(parser)
    RenderTool.init_parser(parser)
    PyrendererSRN.init_parser(parser)
    LossFactory.init_parser(parser)
    Optimizer.init_parser(parser)
    parser.add_argument('--global-seed', type=int, default=124, help='random seed to use. Default=124')
    return parser


def build_dataset(mode: DatasetType, args: Dict[str, Any], volume_data_storage: VolumeDataStorage, volume_evaluator: VolumeEvaluator):
    sampler = PositionSampler.from_dict(args, mode=mode)
    data = WorldSpaceDensityData.from_dict(args, volume_data_storage, mode=mode, volume_evaluator=volume_evaluator, position_sampler=sampler)
    return data


def build_evaluation_helpers(device, dtype, image_evaluator, loss_screen, loss_world, network, visualization_image_size):
    evaluator_train = EvaluateWorld(network, image_evaluator, loss_world, dtype, device)
    evaluator_val = EvaluateWorld(network, image_evaluator, loss_world, dtype, device)
    evaluator_vis = EvaluateScreen(network, image_evaluator, loss_screen, *visualization_image_size, False, False, dtype, device)
    return evaluator_train, evaluator_val, evaluator_vis


def main():
    parser = build_parser()
    args = vars(parser.parse_args())

    dtype = torch.float32

    print(f'[INFO] Device-count: {torch.cuda.device_count()}')

    print('[INFO] Initializing volume data storage.')
    volume_data_storage = VolumeDataStorage.from_dict(args)

    print('[INFO] Initializing rendering tool.')
    render_tool = RenderTool.from_dict(args, device,)
    volume_evaluator = render_tool.get_volume_evaluator()
    if not volume_evaluator.interpolator.grid_resolution_new_behavior:
        volume_evaluator.interpolator.grid_resolution_new_behavior = True
    image_evaluator = render_tool.get_image_evaluator()

    print('[INFO] Creating training dataset.')
    training_data = build_dataset(DatasetType.TRAINING, args, volume_data_storage, volume_evaluator)

    print('[INFO] Creating validation dataset.')
    validation_data = build_dataset(DatasetType.VALIDATION, args, volume_data_storage, volume_evaluator)

    print('[INFO] Creating visualization dataset')
    visualization_data = WorldSpaceVisualizationData.from_dict(args, volume_data_storage, render_tool=render_tool)

    print('[INFO] Building dataset resampler')
    resampler = DatasetResampler.from_dict(args, device=device)

    print('[INFO] Initializing network')
    network = PyrendererSRN.from_dict(
        args,
        member_keys=training_data.ensemble_index,
        dataset_key_times=training_data.timestep_index,
    )
    network.to(device, dtype)

    print('[INFO] Building loss modules')
    loss_screen, loss_world, loss_world_mode = LossFactory.createLosses(args, dtype, device)
    loss_screen.to(device, dtype)
    loss_world.to(device, dtype)

    print('[INFO] Building optimizer')
    optimizer = Optimizer(args, network.parameters(), dtype, device)

    print('[INFO] Creating evaluation helpers')
    evaluator_train, evaluator_val, evaluator_vis = build_evaluation_helpers(device, dtype, image_evaluator, loss_screen, loss_world, network, visualization_data.image_size())
    profile = args['profile']

    def run_training():
        partial_losses = defaultdict(float)
        network.train()
        num_batches = 0
        for data_tuple in training_data.get_dataloader(shuffle=True, drop_last=False):
            num_batches += 1
            data_tuple = utils.toDevice(data_tuple, device)

            def optim_closure():
                optimizer.zero_grad()
                prediction, total, lx = evaluator_train(data_tuple)
                for k, v in lx.items():
                    partial_losses[k] += v
                total.backward()
                # print("Grad latent:", torch.sum(network._time_latent_space.grad.detach()).item())
                # print("Batch, loss:", total.item())
                optimizer.clip_grads()
                return total

            optimizer.step(optim_closure)
        return partial_losses, num_batches

    def run_validation():
        partial_losses = defaultdict(float)
        network.eval()
        num_batches = 0
        with torch.no_grad():
            for j, data_tuple in enumerate(validation_data.get_dataloader(shuffle=False, drop_last=False)):
                num_batches += 1
                data_tuple = utils.toDevice(data_tuple, device)
                prediction, total, lx = evaluator_val(data_tuple)
                for k, v in lx.items():
                    partial_losses[k] += v
        return partial_losses, num_batches

    def run_visualization():
        with torch.no_grad():
            visualizer = Visualizer(
                visualization_data.image_size(),
                visualization_data.num_members(),
                visualization_data.get_dataloader(),
                evaluator_vis, visualization_data.num_tfs(), device
            )
            image = visualizer.draw_image()
        return image

    print('[INFO] Setting up storage directories')

    storage_manager = StorageManager(args, overwrite_output=False)
    storage_manager.print_output_directories()
    storage_manager.make_output_directories()
    storage_manager.store_script_info()
    storage_manager.get_tensorboard_summary()

    epochs_with_save = set(list(range(0, optimizer.num_epochs(), args['output:save_frequency'])) + [optimizer.num_epochs()])
    num_epochs_total = optimizer.num_epochs() + 1
    num_epochs_trained = 0

    # HDF5-output for summaries and export
    with storage_manager.get_hdf5_summary() as hdf5_file:
        storage_manager.initialize_hdf5_storage(
            hdf5_file, num_epochs_total, len(epochs_with_save),
            evaluator_val.loss_names(), network
        )

        print('[INFO] Running training loop')

        start_time = time.time()
        with ExitStack() as stack:
            progress_bar = stack.enter_context(tqdm.tqdm(total=num_epochs_total))
            if profile:
                profiler = build_profiler(stack)
            while num_epochs_trained < num_epochs_total:
                # update network
                if network.start_epoch():
                    optimizer.reset(network.parameters())

                # TRAIN
                partial_losses, num_batches = run_training()
                num_epochs_trained = num_epochs_trained + 1

                storage_manager.update_training_metrics(num_epochs_trained, partial_losses, num_batches, optimizer.get_lr()[0])

                # save checkpoint
                if num_epochs_trained in epochs_with_save:
                    storage_manager.store_torch_checkpoint(num_epochs_trained, network)
                    storage_manager.store_hdf5_checkpoint(network)

                # VALIDATE
                partial_losses, num_batches = run_validation()
                end_time = time.time()
                storage_manager.update_validation_metrics(num_epochs_trained, partial_losses, num_batches, end_time - start_time)

                # VISUALIZE
                if num_epochs_trained in epochs_with_save:
                    with torch.no_grad():
                        # the vis dataset contains one entry per tf-timestep-ensemble
                        # -> concatenate them into one big image
                        image = run_visualization()
                        storage_manager.store_image(num_epochs_trained, image)

                # update training data
                if num_epochs_trained < num_epochs_total and resampler.requires_action(num_epochs_trained):
                    network.eval()
                    resampler.resample_dataset(training_data, volume_evaluator, network)

                # done with this epoch
                if profile:
                    profiler.step()
                final_loss = partial_losses['total'] / max(1, num_batches)
                optimizer.post_epoch(final_loss)
                progress_bar.update(1)
                progress_bar.set_description("Loss: %7.5f" % (final_loss))
                if np.isnan(final_loss):
                    break

    print("Done in", (time.time()-start_time),"seconds")


if __name__== '__main__':
    main()