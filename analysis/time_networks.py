import argparse

import torch
import numpy as np
import os
import imageio
from typing import Tuple, Optional
import pyrenderer

from training.in_out.directories import get_project_base_path, get_output_base_path

BLEND_TO_WHITE_BACKGROUND = True


def convert_image(img):
    out_img = img[0].cpu().detach().numpy()
    if BLEND_TO_WHITE_BACKGROUND:
        rgb = out_img[:3]
        alpha = out_img[3:4]
        white = np.ones_like(rgb)
        out_img = rgb + (1-alpha) * white
    else:
        out_img = out_img[:3]
    out_img *= 255.0
    out_img = out_img.clip(0, 255)
    out_img = np.uint8(out_img)
    out_img = np.moveaxis(out_img, (1, 2, 0), (0, 1, 2))
    return out_img


def evaluate(settings_file: str, volnet_folder: Optional[str], volume_folder:Optional[str], output_folder: str,
             width:int, height:int, grid_size: Tuple[int,int,int], stepsize_world: float=None,
             recursive=True):
    # Load settings file
    print("Load settings from", settings_file)
    image_evaluator = pyrenderer.load_from_json(settings_file)
    assert isinstance(image_evaluator, pyrenderer.ImageEvaluatorSimple)
    default_volume = image_evaluator.volume.volume()
    if stepsize_world is not None:
        assert isinstance(image_evaluator.ray_evaluator, pyrenderer.RayEvaluationSteppingDvr)
        if hasattr(image_evaluator.ray_evaluator, "stepsizeIsObjectSpace"):
            # old renderer version
            image_evaluator.ray_evaluator.stepsizeIsObjectSpace = False
        image_evaluator.ray_evaluator.stepsize = stepsize_world

    timer = pyrenderer.GPUTimer()
    # create grid for sampling the points
    device = torch.device("cuda")
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.linspace(0, 1, grid_size[0], dtype=torch.float32, device=device),
        torch.linspace(0, 1, grid_size[1], dtype=torch.float32, device=device),
        torch.linspace(0, 1, grid_size[2], dtype=torch.float32, device=device))
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    grid_z = grid_z.flatten()
    grid_coords = torch.stack((grid_x, grid_y, grid_z), dim=1)

    print("Save results to", output_folder)
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "timings.csv"), "w") as stats:
        stats.write("File,Rendering (sec),Grid Evaluation (sec)\n")
        print("Render reference")
        img = image_evaluator.render(width, height)
        timer.start()
        img = image_evaluator.render(width, height)
        timer.stop()
        imageio.imwrite(
            os.path.join(output_folder, 'reference.png'),
            convert_image(img))
        time_img = timer.elapsed_milliseconds()

        print("Evaluate points")
        image_evaluator.volume.evaluate(grid_coords)
        timer.start()
        image_evaluator.volume.evaluate(grid_coords)
        timer.stop()
        time_grid = timer.elapsed_milliseconds()

        stats.write("Reference,%.5f,%.5f\n"%(time_img/1000.0, time_grid/1000.0))

        # VOLUMES
        volumes = []
        if volume_folder is not None:
            files = [f for f in os.listdir(volume_folder) if f.endswith('.cvol')]
            for n in sorted(files):
                volumes.append((os.path.join(volume_folder, n), n))
        print("Now render", len(volumes), "volumes")

        volume = image_evaluator.volume
        assert isinstance(volume, pyrenderer.VolumeInterpolationGrid)
        for vpath, vname in volumes:
            v = pyrenderer.Volume(vpath)
            volume.setSource(v, 'tk')
            base_name = os.path.splitext(vname)[0]
            #os.makedirs(os.path.join(output_folder, os.path.split(base_name)[0]), exist_ok=True)

            img = image_evaluator.render(width, height)
            timer.start()
            img = image_evaluator.render(width, height)
            timer.stop()
            imageio.imwrite(
                os.path.join(output_folder, base_name + '.png'),
                convert_image(img))
            time_img = timer.elapsed_milliseconds()

            print("Evaluate points")
            volume.evaluate(grid_coords)
            timer.start()
            volume.evaluate(grid_coords)
            timer.stop()
            time_grid = timer.elapsed_milliseconds()

            stats.write("%s,%.5f,%.5f\n" % (vname, time_img / 1000.0, time_grid / 1000.0))
            stats.flush()

        # NETWORKS
        networks = []
        if volnet_folder is not None:
            files = [f for f in os.listdir(volnet_folder) if f.endswith('.volnet')]
            for n in sorted(files):
                networks.append((os.path.join(volnet_folder, n), n))
        print("Now render", len(networks), "networks")

        volume_network = pyrenderer.VolumeInterpolationNetwork()
        image_evaluator.volume = volume_network
        for npath, nname in networks:
            base_name = os.path.split(nname)[-1]
            print("Render", base_name)
            base_name = os.path.splitext(nname)[0]
            # base_name = base_name.replace("/", "-").replace("\\", "-")
            #os.makedirs(os.path.join(output_folder, os.path.split(base_name)[0]), exist_ok=True)

            srn = pyrenderer.SceneNetwork.load(npath)
            volume_network.set_network(srn)

            img = image_evaluator.render(width, height)
            timer.start()
            img = image_evaluator.render(width, height)
            timer.stop()
            imageio.imwrite(
                os.path.join(output_folder, base_name+'.png'),
                convert_image(img))
            time_img = timer.elapsed_milliseconds()

            print("Evaluate points")
            volume_network.evaluate(grid_coords)
            timer.start()
            volume_network.evaluate(grid_coords)
            timer.stop()
            time_grid = timer.elapsed_milliseconds()

            stats.write("%s,%.5f,%.5f\n" % (nname, time_img / 1000.0, time_grid / 1000.0))
            stats.flush()

    print("Done")


def main2():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-directory', type=str, required=True)
    parser.add_argument('--renderer:settings-file', type=str, required=True)
    args = vars(parser.parse_args())

    WIDTH = 1024
    HEIGHT = 1024
    STEPSIZE_WORLD = 1 / 256

    output_base_dir = os.path.join(args['experiment_directory'], 'stats', 'rendering')
    volnet_base_dir = os.path.join(args['experiment_directory'], 'results', 'volnet')
    runs = sorted(os.listdir(volnet_base_dir))
    for run in runs:
        volnet_dir = os.path.join(volnet_base_dir, run)
        output_dir = os.path.join(output_base_dir, run)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        evaluate(
            args['renderer:settings_file'], volnet_dir, output_dir,
            WIDTH, HEIGHT, STEPSIZE_WORLD
        )


def get_renderings_for_compressor_comparison():
    output_base_dir = '~/Desktop/rendering_data/quality_250/tk'
    gt_dir = '~/Desktop/rendering_data/quality/tk/ground_truth'
    results_base_dir = get_output_base_path()
    multi_grid_dir = os.path.join(results_base_dir, 'ensemble/multi_grid/resampling/12-22-16_32-32-32-32_1-65/results/volnet/run00004')
    multi_core_dir = os.path.join(results_base_dir, 'ensemble/multi_core/resampling/12-22-16_32-32-32-32_1-65/results/volnet/run00004')
    settings_file = os.path.join(get_project_base_path(), "config-files/meteo-ensemble_tk_local-min-max.json")

    def _evaluate(net_dir, vol_dir, target_dir):
        WIDTH = 1024 + 512
        HEIGHT = 1024
        GRIDSIZE = (352, 250, 12)
        STEPSIZE_WORLD = 1 / 256
        evaluate(settings_file, net_dir, vol_dir, target_dir, WIDTH, HEIGHT, GRIDSIZE, STEPSIZE_WORLD)

    _evaluate(None, gt_dir, gt_dir)
    _evaluate(None, multi_core_dir, os.path.join(output_base_dir, 'multi_core'))
    _evaluate(None, multi_grid_dir, os.path.join(output_base_dir, 'multi_grid'))
    _evaluate(None, os.path.join(output_base_dir, 'multi_core'), os.path.join(output_base_dir, 'multi_core'))
    _evaluate(None, os.path.join(output_base_dir, 'multi_grid'), os.path.join(output_base_dir, 'multi_grid'))
    _evaluate(None, os.path.join(output_base_dir, 'sz3'), os.path.join(output_base_dir, 'sz3'))
    _evaluate(None, os.path.join(output_base_dir, 'tthresh'), os.path.join(output_base_dir, 'tthresh'))


if __name__ == '__main__':
    get_renderings_for_compressor_comparison()
