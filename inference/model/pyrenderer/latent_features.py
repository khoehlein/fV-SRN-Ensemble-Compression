import argparse
from typing import Dict, Any, Optional, Union, Tuple, List

import numpy as np
import pyrenderer
import torch

from inference.model.latent_features.init import DefaultInitializer
from inference.model.latent_features.marginal import MarginalLatentFeatures, TemporalFeatureVector, TemporalFeatureGrid, \
    EnsembleFeatureVector, EnsembleFeatureGrid, EnsembleMultiResolutionFeatures, EnsembleMultiGridFeatures, \
    FeatureVector, FeatureGrid, MultiResolutionFeatures
from inference.model.latent_features.regularized_features import RegularizedFeatures


class PyrendererLatentFeatures(MarginalLatentFeatures):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('LatentFeatures')
        prefix = '--network:latent-features:'
        group.add_argument(
            prefix + 'time:num-channels', type=int, default=0,
            help="""
            number of channels for time-related latent-features
            """
        )
        group.add_argument(
            prefix + 'time:mode', type=str, default='vector', choices=['vector', 'grid'],
            help="""
            mode for handling spatial coordinates in time features
            """
        )
        group.add_argument(
            prefix + 'time:grid:resolution', type=str, default=None,
            help="""
            grid size for volumetric temporal features 
            """
        )
        group.add_argument(
            prefix + 'time:key-frames', type=str, default=None,
            help="""
            Key frame specification for temporal features as "min_time:max_time:num_steps" 
            (Default: None, ie. use frames as obtained from the dataset)
            """
        )
        group.add_argument(
            prefix + 'ensemble:num-channels', type=int, default=0,
            help="""
            number of channels for ensemble-related latent-features
            """
        )
        group.add_argument(
            prefix + 'ensemble:mode', type=str, default='vector',
            choices=['vector', 'grid', 'multi-res', 'multi-grid'],
            help="""
            mode for handling spatial coordinates in ensemble features
            """
        )
        group.add_argument(
            prefix + 'ensemble:sparsity-regularization', type=float, default=None,
            help="""
            weight for sparsity regularization
            """
        )
        group.add_argument(
            prefix + 'ensemble:grid:resolution', type=str, default=None,
            help="""
            grid size for volumetric ensemble features 
            """
        )
        group.add_argument(
            prefix + 'ensemble:multi-grid:resolution', type=str, default=None,
            help="""
            grid size for volumetric multi-grid ensemble features 
            """
        )
        group.add_argument(
            prefix + 'ensemble:multi-grid:mixing-mode', type=str, default='normalize',
            choices=['normalize', 'softmax'],
            help="""
            postprocessing method for transforming the mixing features 
            """
        )
        group.add_argument(
            prefix + 'ensemble:multi-grid:num-grids', type=int, default=None,
            help="""
            number of grids for volumetric multi-grid ensemble features 
            """
        )
        group.add_argument(
            prefix + 'ensemble:multi-res:coarsest', type=str, default=None,
            help="""
            grid size for coarsest grid in multi-resolution volumetric ensemble features 
            """
        )
        group.add_argument(
            prefix + 'ensemble:multi-res:finest', type=str, default=None,
            help="""
            grid size for finest grid in multi-resolution volumetric ensemble features 
            """
        )
        group.add_argument(
            prefix + 'ensemble:multi-res:num-levels', type=int, default=2,
            help="""
            number of levels for ensemble-related multi-resolution latent-features
            """
        )
        group.add_argument(
            prefix + 'ensemble:multi-res:table-size', type=int, default=None,
            help="""
            number of table entries for multi-resolution ensemble features
            """
        )
        group.add_argument(
            prefix + 'volume:num-channels', type=int, default=0,
            help="""
            number of channels for position-related latent-features
            """
        )
        group.add_argument(
            prefix + 'volume:mode', type=str, default='vector', choices=['vector', 'grid', 'multi-res', 'multi-grid'],
            help="""
            mode for handling spatial coordinates in volume features
            """
        )
        group.add_argument(
            prefix + 'volume:sparsity-regularization', type=float, default=None,
            help="""
            weight for sparsity regularization
            """
        )
        group.add_argument(
            prefix + 'volume:grid:resolution', type=str, default=None,
            help="""
            grid size for purely volumetric features 
            """
        )
        group.add_argument(
            prefix + 'volume:multi-res:coarsest', type=str, default=None,
            help="""
            grid size for coarsest grid in multi-resolution volumetric features 
            """
        )
        group.add_argument(
            prefix + 'volume:multi-res:finest', type=str, default=None,
            help="""
            grid size for finest grid in multi-resolution volumetric features 
            """
        )
        group.add_argument(
            prefix + 'volume:multi-res:num-levels', type=int, default=2,
            help="""
            number of levels for volumetric multi-resolution latent-features
            """
        )
        group.add_argument(
            prefix + 'volume:multi-res:table-size', type=int, default=None,
            help="""
            number of table entries for multi-resolution volumetric features
            """
        )
        group.add_argument(
            prefix + 'volume:multi-grid:resolution', type=str, default=None,
            help="""
             grid size for volumetric multi-grid features 
             """
        )
        group.add_argument(
            prefix + 'volume:multi-grid:num-grids', type=int, default=None,
            help="""
             number of grids for volumetric multi-grid features 
             """
        )
        group.add_argument(
            prefix + 'volume:multi-grid:mixing-mode', type=str, default='normalize',
            choices=['normalize', 'softmax'],
            help="""
            postprocessing method for transforming the mixing features 
            """
        )

    @classmethod
    def from_dict(
            cls, args: Dict[str, Any],
            member_keys: Optional[List[int]] = None, dataset_key_times: Optional[List[float]] = None
    ):
        prefix = 'network:latent_features:'

        def get_arg(name):
            return args[prefix + name]

        def read_grid_specs(gs: str):
            gs = gs.split(':')
            assert len(gs) in {1, 3}
            if len(gs) == 1:
                gs = [gs[0]] * 3
            out = (int(gs[0]), int(gs[1]), int(gs[2]))
            return out

        temporal_channels = get_arg('time:num_channels')
        if temporal_channels > 0:
            key_time_specs = get_arg('time:key_frames')
            if key_time_specs is None:
                assert dataset_key_times is not None
                key_times = torch.tensor(dataset_key_times)
            else:
                min_time, max_time, num_steps = key_time_specs.split(':')
                key_times = torch.linspace(float(min_time), float(max_time), int(num_steps))
            mode = get_arg('time:mode')
            if mode == 'vector':
                temporal_features = TemporalFeatureVector(key_times, temporal_channels)
            elif mode == 'grid':
                grid_specs = get_arg('time:grid:resolution')
                assert grid_specs is not None
                grid_size = read_grid_specs(grid_specs)
                temporal_features = TemporalFeatureGrid(key_times, temporal_channels, grid_size)
            else:
                raise NotImplementedError()
        else:
            temporal_features = None

        ensemble_channels = get_arg('ensemble:num_channels')
        if ensemble_channels > 0:
            assert member_keys is not None, \
                '[ERROR] member keys must be provided if ensemble features are to be used.'
            mode = get_arg('ensemble:mode')
            if mode == 'vector':
                ensemble_features = EnsembleFeatureVector(member_keys, ensemble_channels)
            elif mode == 'grid':
                grid_specs = get_arg('ensemble:grid:resolution')
                assert grid_specs is not None
                grid_size = read_grid_specs(grid_specs)
                ensemble_features = EnsembleFeatureGrid(
                    member_keys, ensemble_channels, grid_size,
                )
            elif mode == 'multi-res':
                coarsest = get_arg('ensemble:multi_res:coarsest')
                assert coarsest is not None
                coarsest = read_grid_specs(coarsest)
                finest = get_arg('ensemble:multi_res:finest')
                assert finest is not None
                finest = read_grid_specs(finest)
                num_levels = get_arg('ensemble:multi_res:num_levels')
                assert ensemble_channels % num_levels == 0
                t = get_arg('ensemble:multi_res:table_size')
                assert t is not None
                ensemble_features = EnsembleMultiResolutionFeatures(member_keys, ensemble_channels, coarsest, finest, num_levels, t)
            elif mode == 'multi-grid':
                grid_specs = get_arg('ensemble:multi_grid:resolution')
                assert grid_specs is not None
                grid_size = read_grid_specs(grid_specs)
                num_grids = get_arg('ensemble:multi_grid:num_grids')
                ensemble_features = EnsembleMultiGridFeatures(
                    member_keys, ensemble_channels, grid_size, num_grids,
                    mixing_mode=get_arg('ensemble:multi_grid:mixing_mode'),
                )
            else:
                raise NotImplementedError()
            sparsity_weight = get_arg('ensemble:sparsity_regularization')
            if sparsity_weight is not None:
                ensemble_features = RegularizedFeatures(ensemble_features, sparsity_weight)
        else:
            ensemble_features = None

        volumetric_channels = get_arg('volume:num_channels')
        if volumetric_channels > 0:
            mode = get_arg('volume:mode')
            initializer = DefaultInitializer()
            if mode == 'vector':
                volumetric_features = FeatureVector.from_initializer(initializer, volumetric_channels)
            elif mode == 'grid':
                grid_specs = get_arg('volume:grid:resolution')
                assert grid_specs is not None
                grid_size = read_grid_specs(grid_specs)
                volumetric_features = FeatureGrid.from_initializer(initializer, grid_size, volumetric_channels)
            elif mode == 'multi-res':
                coarsest = get_arg('ensemble:multi_res:coarsest')
                assert coarsest is not None
                coarsest = read_grid_specs(coarsest)
                finest = get_arg('ensemble:multi_res:finest')
                assert finest is not None
                finest = read_grid_specs(finest)
                num_levels = get_arg('volume:multi_res:num_levels')
                assert volumetric_channels % num_levels == 0
                t = get_arg('volume:multi_res:table_size')
                assert t is not None
                volumetric_features = MultiResolutionFeatures.from_initializer(initializer, coarsest, finest, num_levels, t, volumetric_channels)
            elif mode == 'multi-grid':
                grid_specs = get_arg('volume:multi_grid:resolution')
                assert grid_specs is not None
                grid_size = read_grid_specs(grid_specs)
                num_grids = get_arg('volume:multi_grid:num_grids')
                volumetric_features = EnsembleMultiGridFeatures(
                    member_keys, volumetric_channels, grid_size, num_grids,
                    mixing_mode=get_arg('volume:multi_grid:mixing_mode')
                )
            else:
                raise NotImplementedError()
            sparsity_weight = get_arg('volume:sparsity_regularization')
            if sparsity_weight is not None:
                volumetric_features = RegularizedFeatures(volumetric_features, sparsity_weight)
        else:
            volumetric_features = None
        if temporal_features is not None or ensemble_features is not None or volumetric_features is not None:
            return cls(temporal_features, ensemble_features, volumetric_features)
        else:
            return None

    def export_to_pyrenderer(
            self,
            grid_encoding,
            network: pyrenderer.SceneNetwork,
            time=None, ensemble=None
    ) -> Tuple[pyrenderer.SceneNetwork, float, int]:
        """
        Fills the latent grid of the given network.
        If the latent grid has a channel count that is not a multiple of 16, the channels are padded
         to the next multiple of 16. This pad count is returned, and has to be applied as well to the network.
        Returns: the modified network, the encoding error, extra padding.
        """
        if self.uses_linear_features():
            raise RuntimeError('[ERROR] Use of linear features is not supported in pyrenderer export!')
        encoding_error = 0
        encoding_error_count = 0
        padding = 0

        def pad_grid(grid: torch.Tensor, axis=1):
            C = grid.shape[axis]
            C16 = (C + 15) // 16 * 16
            if C16 != C:
                # we need to pad
                new_shape = list(grid.shape)
                new_shape[axis] = C16-C
                p = torch.zeros(new_shape, dtype=grid.dtype, device=grid.device)
                return torch.cat((grid, p), dim=axis), C16-C
            else:
                return grid, 0

        if self.uses_positions():
            if self.uses_time() or self.uses_member():
                time_key_frames = self.temporal_features.get_time_key_frames().tolist() if self.uses_time() else [0]
                if len(time_key_frames) > 1:
                    delta = time_key_frames[0] - time_key_frames[1]
                    expected = delta * np.arange(len(time_key_frames)) + time_key_frames[0]
                    assert np.all(np.array(time_key_frames) == expected), \
                        '[ERROR] Pyrenderer doesnot support irregular time grids.'
                ensemble_keys = self.ensemble_features.key_mapping if self.uses_member() else [0]
                ensemble_keys = list(sorted(ensemble_keys.keys()))
                if len(ensemble_keys) > 1:
                    expected = np.arange(len(ensemble_keys)) + ensemble_keys[0]
                    assert np.all(np.array(ensemble_keys) == expected), \
                        '[ERROR] Pyrenderer does not support irregular member keys.'

                if ensemble is not None:
                    # only export a single ensemble
                    ensemble_min = 0
                    ensemble_num = 1
                else:
                    ensemble_min = min(ensemble_keys)
                    ensemble_num = self.ensemble_features.num_members() if self.uses_member() else 0

                grid_info = pyrenderer.SceneNetwork.LatentGridTimeAndEnsemble(
                    time_min=time_key_frames[0],
                    time_num=self.temporal_features.num_key_times() if self.uses_time() else 0,
                    time_step=time_key_frames[1] - time_key_frames[0] if len(time_key_frames) > 1 else 1,
                    ensemble_min=ensemble_min,
                    ensemble_num=ensemble_num)

                if self.uses_time():
                    grid = self.temporal_features.get_grid().unsqueeze(0)
                    grid, padding = pad_grid(grid)
                    for i in range(self.temporal_features.num_key_times()):
                        e = grid_info.set_time_grid_from_torch(i, grid[i:i + 1], grid_encoding)
                        encoding_error += e
                        encoding_error_count += 1
                if self.uses_member():
                    if ensemble is not None:
                        grid = self.ensemble_features.get_grid(ensemble).unsqueeze(0)
                        grid, padding = pad_grid(grid)
                        e = grid_info.set_ensemble_grid_from_torch(0, grid, grid_encoding)
                        encoding_error += e
                        encoding_error_count += 1
                    else:
                        for i in range(self.ensemble_features.num_members()):
                            grid = self.ensemble_features.get_grid(i).unsqueeze(0)
                            grid, padding = pad_grid(grid)
                            e = grid_info.set_ensemble_grid_from_torch(i, grid, grid_encoding)
                            encoding_error += e
                            encoding_error_count += 1
                network.latent_grid = grid_info
            else:
                grid = self.volumetric_features.get_grid().unsqueeze(0)
                grid, padding = pad_grid(grid)
                grid_info = pyrenderer.SceneNetwork.LatentGridTimeAndEnsemble(
                    time_min=0, time_num=1, time_step=1,
                    ensemble_min=0, ensemble_num=0)
                e = grid_info.set_time_grid_from_torch(0, grid, grid_encoding)
                encoding_error += e
                encoding_error_count += 1
                network.latent_grid = grid_info
            if not network.latent_grid.is_valid():
                raise RuntimeError('[ERROR] Exported latent grid is invalid')

        out = encoding_error / encoding_error_count if encoding_error_count > 0 else 0
        return network, out, padding
