import argparse
import io
import json
import os
import shutil
import subprocess
import sys

import h5py
import imageio
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class StorageManager(object):

    class CheckpointKey(object):
        EPOCH = 'epoch'
        PARAMETERS = 'parameters'
        MODEL = 'model'

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser, base_directory):
        parser_group = parser.add_argument_group("Output")
        prefix = '--output:'
        parser.add_argument(prefix + 'base-dir', type=str, default=base_directory)
        parser_group.add_argument(prefix + 'log-dir', type=str, default='results/log',
                                  help='directory for tensorboard logs')
        parser_group.add_argument(prefix + 'checkpoint-dir', type=str, default='results/model',
                                  help='Output directory for the checkpoints')
        parser_group.add_argument(prefix + 'hdf5-dir', type=str, default='results/hdf5',
                                  help='Output directory for the hdf5 summary files')
        parser_group.add_argument(prefix + 'experiment-name', type=str, default=None,
                                  help='Output name. If not specified, use the next available index')
        parser_group.add_argument(prefix + 'save-frequency', type=int, default=10,
                                  help='Every that many epochs, a checkpoint is saved')
        parser_group.add_argument('--profile', action='store_true')

    def __init__(self, opt, overwrite_output=False):
        self.opt = opt
        self.opt.update({
            key: os.path.join(opt['output:base_dir'], opt[key])
            for key in ['output:log_dir', 'output:checkpoint_dir', 'output:hdf5_dir']
        })
        self.overwrite_output = overwrite_output

    def print_output_directories(self):
        print("Model directory:", self.opt['output:checkpoint_dir'])
        print("Log directory:", self.opt['output:log_dir'])
        print("HDF5 directory:", self.opt['output:hdf5_dir'])

    def _find_next_run_number(self, folder):
        if not os.path.exists(folder): return 0
        files = os.listdir(folder)
        files = sorted([f for f in files if f.startswith('run')])
        if len(files) == 0:
            return 0
        return int(files[-1][3:])

    def make_output_directories(self):
        opt = self.opt
        if opt['output:experiment_name'] is None:
            nextRunNumber = max(self._find_next_run_number(opt['output:log_dir']),
                                self._find_next_run_number(opt['output:checkpoint_dir'])) + 1
            print('Current run: %05d' % nextRunNumber)
            runName = 'run%05d' % nextRunNumber
        else:
            runName = opt['output:experiment_name']
            self.overwrite_output = True
        self.log_dir = os.path.join(opt['output:log_dir'], runName)
        self.checkpoint_dir = os.path.join(opt['output:checkpoint_dir'], runName)
        self.hdf5_file = os.path.join(opt['output:hdf5_dir'], runName + ".hdf5")
        if self.overwrite_output and (os.path.exists(self.log_dir) or os.path.exists(self.checkpoint_dir) or os.path.exists(self.hdf5_file)):
            print(f"Warning: Overwriting previous run with name {runName}")
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=self.overwrite_output)
        os.makedirs(self.checkpoint_dir, exist_ok=self.overwrite_output)
        os.makedirs(opt['output:hdf5_dir'], exist_ok=True)

    def store_script_info(self):
        with open(os.path.join(self.checkpoint_dir, 'args.json'), "w") as f:
            json.dump(self.opt, f, indent=4, sort_keys=True)
        with open(os.path.join(self.checkpoint_dir, 'cmd.txt'), "w") as f:
            import shlex
            f.write('cd "%s"\n' % os.getcwd())
            f.write(' '.join(shlex.quote(x) for x in sys.argv) + "\n")

    def opt_string(self):
        return str(self.opt)

    def get_tensorboard_summary(self):
        self.writer = SummaryWriter(self.log_dir)
        self.writer.add_text('info', self.opt_string(), 0)
        return self.writer

    def get_hdf5_summary(self):
        hdf5_file = h5py.File(self.hdf5_file, 'w')
        for k, v in self.opt.items():
            try:
                hdf5_file.attrs[k] = v
            except TypeError as ex:
                print(f'[WARN] Exception {ex} while saving attribute {k} = {v}')
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            hdf5_file.attrs['git'] = git_commit
            print("[INFO] Git commit:", git_commit)
        except:
            print("[WARN] Storage manager was unable to get git commit.")
        return hdf5_file

    @staticmethod
    def save_network(network):
        weights_bytes = io.BytesIO()
        torch.save(network.state_dict(), weights_bytes)
        return np.void(weights_bytes.getbuffer())

    def initialize_hdf5_storage(self, hdf5_file, num_epochs, num_epochs_with_save, loss_names, network):
        self.times = hdf5_file.create_dataset("times", (num_epochs,), dtype=np.float32)
        self.losses = dict([
            (name, hdf5_file.create_dataset(name, (num_epochs,), dtype=np.float32))
            for name in loss_names
        ])
        self.epochs = hdf5_file.create_dataset("epochs", (num_epochs,), dtype=int)
        self.weights = hdf5_file.create_dataset(
            "weights",
            (num_epochs_with_save, self.save_network(network).shape[0]),
            dtype=np.dtype('V1'))
        self.export_weights_counter = 0
        self.export_stats_counter = 0
        return self

    def _check_for_attribute(self, *attrs):
        for attr in attrs:
            if not hasattr(self, attr):
                raise AttributeError(f'[ERROR] StorageManager does not have {attr} initialized yet!')

    def store_torch_checkpoint(self, epoch, network):
        self._check_for_attribute('checkpoint_dir')
        model_out_path = os.path.join(self.checkpoint_dir, "model_epoch_{}.pth".format(epoch if epoch >= 0 else "init"))
        state = {
            StorageManager.CheckpointKey.EPOCH: epoch + 1,
            StorageManager.CheckpointKey.MODEL: network,
            StorageManager.CheckpointKey.PARAMETERS: self.opt}
        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def store_hdf5_checkpoint(self, network):
        self._check_for_attribute('weights')
        self.weights[self.export_weights_counter, :] = self.save_network(network)
        self.export_weights_counter = self.export_weights_counter + 1

    def update_training_metrics(self, epoch, losses, num_batches, lr):
        self._check_for_attribute('writer')
        for k, v in losses.items():
            self.writer.add_scalar('train/%s' % k, v / num_batches, epoch)
        self.writer.add_scalar('train/lr', lr, epoch)

    def update_validation_metrics(self, epoch, losses, num_batches, run_time):
        self._check_for_attribute('writer', 'times', 'losses', 'epochs')
        for k, v in losses.items():
            self.writer.add_scalar('val/%s' % k, v / num_batches, epoch)
        self.times[self.export_stats_counter] = run_time
        for k, v in losses.items():
            self.losses[k][self.export_stats_counter] = v / num_batches
        self.epochs[self.export_stats_counter] = epoch
        self.export_stats_counter += 1

    def store_image(self, epoch, image):
        self._check_for_attribute('writer')
        img_np = np.array(image)
        imageio.imwrite(os.path.join(self.log_dir, ("e%d.png" % epoch) if epoch >= 0 else "eInit.png"), img_np)
        self.writer.add_image('vis', np.moveaxis(img_np, (0, 1, 2), (1, 2, 0)), epoch)
