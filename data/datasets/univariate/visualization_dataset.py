import argparse
from itertools import product

import numpy as np
import tqdm
from torch.utils.data import Dataset, DataLoader

from common.mathparser import BigFloat
from common.utils import parse_slice_string
from data.datasets._cache import _MCCache
from inference.volume import RenderTool


class WorldSpaceVisualizationData(Dataset):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        prefix = '--world-vis-data:'
        group = parser.add_argument_group('WorldSpaceVisualizationData')
        group.add_argument(prefix + 'ensemble-index-slice', type=str, default=':', help="""
        Slice of the available ensemble index range to use for visualization.
        
        Examples: ":", "0::2", "1::2", "0:10:2" (Default: ":")
        """)
        group.add_argument(prefix + 'timestep-index-slice', type=str, default=':', help="""
        Slice of the available timestep index range to use for data generation

        Examples: ":", "0::2", "1::2", "0:10:2" (Default: ":")
        """)
        group.add_argument(prefix + 'resolution', type=int, default=256, help="""
        The resolution of the images / views for visualization in X and Y direction.
        """)
        group.add_argument(prefix + 'resolution:X', type=int, default=None, help="""
        The resolution of the images / views for visualization in X direction.
        Overwrites common X-Y setting.
        """)
        group.add_argument(prefix + 'resolution:Y', type=int, default=None, help="""
        The resolution of the images / views for visualization in Y direction.
        Overwrites common X-Y setting.
        """)
        group.add_argument(prefix + 'step-size', type=BigFloat, default=0.005, help="""
        The stepsize for raytracing during visualization in world space.
        Arbitrary math expressions like "1/256" are supported
        """)
        group.add_argument(prefix + 'refinements', type=int, default=None, help="""
        The number of refinement iterations for monte-carlo traced images.
        Default: use from training data
        """)
        group.add_argument(prefix + 'enable-image-caching', action='store_true', help="""
        If specified, rendered images are cached.
        This is especially useful for monte-carlo renderings.""")
        group.set_defaults(**{(prefix + 'enable_image_caching'): False})

    @classmethod
    def from_dict(cls, args, volume_data_storage, render_tool=None):
        prefix = 'world_vis_data:'

        def get_argument(option):
            return args[prefix + option]

        # get resolution tuple
        resolution = [get_argument('resolution')] * 2
        res_x = get_argument('resolution:X')
        if res_x is not None:
            resolution[0] = res_x
        res_y = get_argument('resolution:Y')
        if res_y is not None:
            resolution[1] = res_y
        resolution = tuple(resolution)

        # get remaining arguments
        ensemble_index_slice = get_argument('ensemble_index_slice')
        timestep_index_slice = get_argument('timestep_index_slice')
        step_size = get_argument('step_size')
        num_refinements = get_argument('refinements')
        cache_enabled = get_argument('enable_image_caching')

        return cls(
            volume_data_storage, resolution, step_size=step_size, num_refinements=num_refinements,
            ensemble_index_slice=ensemble_index_slice, timestep_index_slice=timestep_index_slice,
            cache_enabled=cache_enabled, render_tool=render_tool
        )

    def __init__(
            self,
            volume_data_storage, resolution,
            step_size=5.e-3, num_refinements=None,
            ensemble_index_slice=None, timestep_index_slice=None,
            num_views=1, cache_enabled=True, render_tool=None
    ):
        assert num_views == 1,'[ERROR] WorldSpaceVisualizationData currently supports only single-view setting'
        self.num_views = num_views
        self.volume_data_storage = volume_data_storage
        if type(resolution) not in [tuple, list]:
            int_resolution = int(resolution)
            assert int_resolution == resolution, '[ERROR] Resolution cannot be interpreted as int'
            resolution = [int_resolution] * 2
        self.resolution = tuple(resolution)
        self.step_size = step_size
        if num_refinements is None:
            num_refinements = 0
        self.num_refinements = num_refinements
        if ensemble_index_slice is None:
            ensemble_index_slice = ':'
        self.ensemble_index = volume_data_storage.ensemble_index[slice(*parse_slice_string(ensemble_index_slice))]
        if timestep_index_slice is None:
            timestep_index_slice = ':'
        self.timestep_index = volume_data_storage.timestep_index[slice(*parse_slice_string(timestep_index_slice))]
        self._cache_enabled = cache_enabled
        self.cache = None
        self._reset_data()
        if render_tool is not None:
            self.build_dataset(render_tool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def _reset_data(self):
        self.data = {}

    def _update_data(self, dataset):
        self.data.update({i: x for i, x in enumerate(dataset)})

    def _initialize_cache(self, settings_file):
        self.cache = _MCCache(settings_file)

    def build_dataset(self, render_tool: RenderTool):
        self._reset_data()
        if self._cache_enabled:
            settings_file = render_tool.get_settings_file()
            if settings_file is not None:
                self._initialize_cache(settings_file)
        dataset = []
        with tqdm.tqdm(self.num_timesteps() * self.num_members() * self.num_views) as iteration_bar:
            iteration_bar.set_description("Render")
            for (timestep_index, timestep), (ensemble_index, ensemble) in product(enumerate(self.timestep_index), enumerate(self.ensemble_index)):
                volume_data = self.volume_data_storage.load_volume(timestep=timestep, ensemble=ensemble, index_access=False)
                image_evaluator = render_tool.set_source(volume_data).get_image_evaluator()
                image_evaluator.camera.pitchYawDistance.value = render_tool.default_camera_pitch_yaw_distance()
                camera_parameters = image_evaluator.camera.get_parameters()
                assert camera_parameters.shape[0] == 1  # no batches
                cached_images = None
                if self.cache is not None:
                    # TODO: Cache does currently not work due to switch from integer to tuple resolution convention!
                    cached_images = self.cache.query(0, timestep, ensemble, 1, self.resolution, self.num_refinements)
                if cached_images is None:
                    img = self._render_and_refine(image_evaluator).cpu().numpy()
                    if self.cache is not None:
                        self._rebuild_cache(img)
                else:
                    img = cached_images
                tf_index_data, timestep_index_data, ensemble_index_data = self._build_index_data(0, timestep_index, ensemble_index)
                dataset.append((
                    camera_parameters.cpu().numpy()[0], img[0, ...],
                    tf_index_data, timestep_index_data, ensemble_index_data,
                    self.step_size
                ))
                iteration_bar.update(1)
        self._update_data(dataset)
        render_tool.restore_defaults()
        return self

    def _rebuild_cache(self, img):
        self.cache.put(img)

    def _render_and_refine(self, image_evaluator):
        img = image_evaluator.render(*self.resolution)
        if self.num_refinements > 0:
            for _ in tqdm.trange(self.num_refinements, desc='Refine'):
                img = image_evaluator.refine(*self.resolution, img)
        # tonemapping
        img = image_evaluator.extract_color(img)
        return img

    def _build_index_data(self, tf_index, timestep_index, ensemble_index):
        tf_index_data = np.full((1,), tf_index, dtype=np.int32)
        timestep_index_data = np.full((1,), timestep_index, dtype=np.float32)
        ensemble_index_data = np.full((1,), ensemble_index, dtype=np.float32)
        return tf_index_data, timestep_index_data, ensemble_index_data

    def num_timesteps(self):
        return len(self.timestep_index)

    def num_members(self):
        return len(self.ensemble_index)

    def num_ensembles(self):
        return self.num_members()

    def image_size(self):
        return self.resolution

    def get_dataloader(self):
        return DataLoader(self, batch_size=1, shuffle=False, drop_last=False)

    def num_tfs(self):
        return 1

