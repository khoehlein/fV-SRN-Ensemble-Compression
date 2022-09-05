import argparse

import torch

from inference.model.pyrenderer import PyrendererSRN
from training.in_out.storage_manager import StorageManager


class MultiGridModelLoader(object):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('multi-grid-loader')
        group.add_argument('--multi-grid-loader:pth-path', type=str, required=True, help="path to pth file to load")

    @classmethod
    def from_dict(cls, args):
        return cls(args['multi_grid_loader:pth_path'])

    def __init__(self, pth_path):
        self.pth_path = pth_path

    def load_model_and_parameters(self):
        checkpoint = torch.load(self.pth_path, map_location='cpu')
        model = checkpoint[StorageManager.CheckpointKey.MODEL]
        args = checkpoint[StorageManager.CheckpointKey.PARAMETERS]
        return model, args

    def build_model(self, args, member_keys=None, dataset_time_keys=None, output_mode=None):
        old_model, old_args = self.load_model_and_parameters()
        self.update_args(args, old_args)
        model = PyrendererSRN.from_dict(args, member_keys=member_keys, dataset_key_times=dataset_time_keys, output_mode=output_mode)
        self.update_core_model(model, old_model)
        return model, args

    def update_args(self, new_args, old_args):
        additional_args = {k: p for k, p in old_args.items() if k not in new_args}
        new_args.update(additional_args)
        return new_args

    def update_core_model(self, new_model, old_model: PyrendererSRN):
        core_model = old_model.core_network
        for p in core_model.parameters():
            p.requires_grad = False
        new_model.core_network = core_model
        return new_model


