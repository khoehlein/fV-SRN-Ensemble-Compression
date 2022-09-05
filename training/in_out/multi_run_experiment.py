import datetime
import json
import math
import os.path
import subprocess
import time
from itertools import product, chain
from typing import Dict, Any, List

import numpy as np

from external.utils.automation.devices import DeviceManager


class MultiRunExperiment(object):

    def __init__(
            self,
            interpreter_path: str,
            script_path: str,
            working_dir: str,
            log_dir: str,
            create_log_dir=False,
            verbose=True
    ):
        self.interpreter_path = interpreter_path
        self.script_path = script_path
        self.working_dir = working_dir
        self.log_dir = log_dir
        if (log_dir is not None) and (not os.path.isdir(log_dir)):
            if create_log_dir:
                os.makedirs(log_dir)
            else:
                raise RuntimeError(f'[ERROR] Log directory {log_dir} does not exist!')
        self.verbose = verbose

    def process_parameters(self, kwargs: Dict[str, Any] = None,  flags: List[str] = None, randomize=True):
        constant_params = {key: val for key, val in kwargs.items() if not isinstance(val, list)}
        varying_parameters = [list(product([key], vals)) for key, vals in kwargs.items() if key not in constant_params]
        if flags is None:
            flags = []
        configs = list(product(*varying_parameters))
        if self.verbose:
            print(f'[INFO] Processing {len(configs)} script configurations!')
        if randomize:
            np.random.shuffle(configs)
        num_configs = len(configs)
        durations = []
        for i, config in enumerate(configs):
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            if self.verbose:
                print(f'[INFO] Starting new configuration run (timestamp: {timestamp}).')
            time_start = time.time()
            full_config = {**constant_params, **dict(config)}
            self._run_config(full_config, flags, timestamp)
            time_end = time.time()
            duration = (time_end - time_start)
            durations.append(duration)
            if self.verbose:
                print(f'[INFO] Finished configuration run. Duration: {duration}')
                if i + 1 < num_configs:
                    self._print_time_remaining(durations, num_configs - i)

    def _run_config(self, config: Dict[str, Any], flags: List[str], timestamp: str):

        device_manager = DeviceManager()
        free_devices = device_manager.find_free_devices(num_devices=1, force_on_single=True)
        print(f'[INFO] Free devices: {free_devices}')

        execution_command = [self.interpreter_path, self.script_path]
        execution_command += [f'{arg}' for arg in chain.from_iterable(config.items())]
        execution_command += flags

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = free_devices[0]

        log_file = None
        if self.log_dir is not None:
            output_folder = os.path.join(self.log_dir, timestamp)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            with open(os.path.join(output_folder, f'{timestamp}_config.json'), 'w') as f:
                json.dump(config, f, indent=4, sort_keys=True)
            with open(os.path.join(output_folder, f'{timestamp}_flags.json'), 'w') as f:
                json.dump({'flags': flags}, f, indent=4, sort_keys=True)
            log_file = open(os.path.join(output_folder, f'{timestamp}_output.log'), 'w')
            stdout = log_file
            stderr = log_file
        else:
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE
        process = subprocess.Popen(
            execution_command, cwd=self.working_dir,
            stdout=stdout, stderr=stderr, env=env,
        )
        outputs, errors = process.communicate()
        if log_file is not None:
            log_file.close()

    @staticmethod
    def _print_time_remaining(durations, num_remaining):
        mu = np.mean(durations) * num_remaining
        if len(durations) > 1:
            std = math.sqrt(num_remaining) * np.std(durations, ddof=1)
            std_text = f' +- {std}'
        else:
            std_text = ''
        print(f'[INFO] Time remaining: {mu}{std_text} seconds')


if __name__ == '__main__':
    pass
    # experiment = MultiRunExperiment(
    #     'something', 'else', 'who_cares'
    # )
    # experiment.process_parameters(
    #     {
    #         'a': 1,
    #         'b': 8,
    #         'c': [3, 4],
    #         'd': [5, 6, 7],
    #     }
    # )