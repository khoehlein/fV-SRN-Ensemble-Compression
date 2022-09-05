import argparse

import torch

from inference.model.pyrenderer import PyrendererSRN
from inference.model.pyrenderer.core_network.core_network import PyrendererMultiCoreNetwork
from training.in_out.storage_manager import StorageManager


class MultiCoreModelLoader(object):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('multi-core-loader')
        group.add_argument('--multi-core-loader:pth-path', type=str, required=True, help="path to pth file to load")

    @classmethod
    def from_dict(cls, args):
        return cls(args['multi_core_loader:pth_path'])

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
        assert isinstance(model.core_network, PyrendererMultiCoreNetwork)
        self.update_latent_features(model, old_model)
        return model, args

    def update_args(self, new_args, old_args):
        additional_args = {k: p for k, p in old_args.items() if k not in new_args}
        new_args.update(additional_args)
        return new_args

    def update_latent_features(self, new_model, old_model: PyrendererSRN):
        latent_features = old_model.latent_features.volumetric_features
        for p in latent_features.parameters():
            p.requires_grad = False
        new_model.latent_features.volumetric_features = latent_features
        return new_model


