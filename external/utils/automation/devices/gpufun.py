import subprocess
import os
from enum import Enum
import torch

# adapted from Christian Reinbold


tolerable_processes = [
    '/usr/lib/xorg/Xorg'
]


class DeviceState(Enum):
    UNKNOWN = 'unknown'
    FREE = 'free'
    OCCUPIED = 'occupied'


class NvidiaSMIStateReader(object):

    def __init__(self):
        self.smi_str = self.call_nvidia_smi()
        sep_iter = iter(i for i, line in enumerate(self.smi_str) if line.startswith('='))
        self._start_gpu_block = next(sep_iter)
        self._start_process_block = next(sep_iter)
        self.available_devices = set(self._yield_available_devices())
        self.occupied_devices = set(self._yield_occupied_devices())

    def call_nvidia_smi(self, print_output=False):
        result_bstr = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
        result_str = result_bstr.stdout.decode('utf-8')
        lines = [s[1:-1].strip() for s in result_str.split('\n')]
        if len(lines) < 4:
            raise Exception('[ERROR] Got unexpected program response (likely CUDA driver error)')
        if print_output:
            print('[INFO] NVIDIA-SMI Output:')
            print('\n'.join(lines))
        return lines

    def _yield_available_devices(self):
        for line in self.smi_str[self._start_gpu_block:self._start_process_block]:
            if len(line) == 0:
                break
            entry = line.split()[0].strip()
            if entry.isnumeric():
                yield entry
            else:
                continue

    def _yield_occupied_devices(self):
        for line in self.smi_str[self._start_process_block:]:
            if len(line) == 0:
                break
            entries = [e.strip() for e in line.split()]
            if entries[0].isnumeric() and entries[5] not in tolerable_processes:
                yield entries[0]
            else:
                continue


class DeviceManager(object):

    def __init__(self):
        self.device_state = None
        self.visible_devices = None
        self._query_devices()

    def _query_devices(self):
        state_reader = NvidiaSMIStateReader()
        self.device_state = {
            key: DeviceState.OCCUPIED if key in state_reader.occupied_devices else DeviceState.FREE
            for key in state_reader.available_devices
        }

    def set_visible_devices(self, *devices):
        if len(devices) == 0:
            raise Exception()
        inputs = set()
        for device_string in devices:
            assert type(device_string) == str
            splitted_devices = devices[0].split(',')
            for device in splitted_devices:
                assert device in self.device_state
            inputs = inputs.union(set(splitted_devices))
        self.visible_devices = list(inputs)
        self._set_environment_variable()
        return self.visible_devices

    def set_free_devices_as_visible(self, num_devices=1, force_on_single=False):
        num_available_devices = len(self.device_state)
        free_devices = [key for key, state in self.device_state.items() if state == DeviceState.FREE]
        if num_available_devices < num_devices:
            raise RuntimeError('[ERROR] Not enough CUDA devices available.')
        elif num_available_devices == 1:
            self._handle_single_gpu_setting(free_devices, force_on_single)
        else:
            self._handle_multi_gpu_setting(free_devices, num_devices)
        self._set_environment_variable()
        return self.visible_devices

    def _handle_single_gpu_setting(self, free_devices, force_on_single):
        print('[INFO] Detected single GPU setting.')
        if len(free_devices) == 1 or force_on_single:
            self.visible_devices = [next(iter(self.device_state.keys()))]
        else:
            raise RuntimeError('[ERROR] Not enough free CUDA devices available.')

    def _handle_multi_gpu_setting(self, free_devices, num_devices):
        print('[INFO] Detected multi-GPU setting.')
        if len(free_devices) >= num_devices:
            self.visible_devices = free_devices[:num_devices]
        else:
            raise RuntimeError('[ERROR] Not enough free CUDA devices available.')

    def _set_environment_variable(self):
        devices_str = ','.join(self.visible_devices)
        os.environ['CUDA_VISIBLE_DEVICES'] = devices_str

    def get_torch_devices(self):
        return [torch.device(f'cuda:{i}') for i in range(len(self.visible_devices))]


# def iterate_total_devices(smi_strings):
#     proc_line_idx = [s.startswith('Processes:') for s in smi_strings].index(True)
#     for line in smi_strings[8:proc_line_idx - 3:4]:
#         yield line.split()[0].strip()
#
#
# def iterate_occupied_devices(smi_strings):
#     proc_line_idx = [s.startswith('Processes:') for s in smi_strings].index(True)
#     for line in smi_strings[proc_line_idx + 4:-2]:
#         if line.startswith('No running processes found'):
#             break
#         else:
#             entries = [e.strip() for e in line.split()]
#             process_name = entries[5]
#             if process_name in admissible_processes:
#                 continue
#             yield line.split()[0].strip()


# def find_free_cuda_devices():
#     smi_str = call_nvidia_smi()
#     sep_iter = iter(i for i, line in enumerate(smi_str) if line.startswith('='))
#     start_gpu_block = next(sep_iter)
#     start_process_block = next(sep_iter)
#     all_devices = set(read_gpu_block(smi_str[start_gpu_block:start_process_block]))
#     occupied_devices = set(read_process_block(smi_str[start_process_block:]))
#     return all_devices, occupied_devices
#
#
# def read_gpu_block(block):
#     for line in block:
#         if len(line) == 0:
#             break
#         entry = line.split()[0].strip()
#         if entry.isnumeric():
#             yield entry
#         else:
#             continue
#
#
# def read_process_block(block):
#     for line in block:
#         if len(line) == 0:
#             break
#         entries = [e.strip() for e in line.split()]
#         if entries[0].isnumeric() and entries[5] not in tolerable_processes:
#             yield entries[0]
#         else:
#             continue


# def set_free_devices_as_visible(num_devices=1, force_on_single=False):
#     all_devices, occupied_devices = find_free_cuda_devices()
#     num_available_devices = len(all_devices)
#     free_devices = list(all_devices - occupied_devices)
#     if num_available_devices < num_devices:
#         raise RuntimeError('[ERROR] Not enough CUDA devices available.')
#     elif num_available_devices == 1:
#         print('[INFO] Detected single GPU setting.')
#         if len(free_devices) == 1 or force_on_single:
#             visible_devices = list(all_devices)[0]
#         else:
#             raise RuntimeError('[ERROR] Not enough free CUDA devices available.')
#     else:
#         print('[INFO] Detected multi-GPU setting.')
#         if len(free_devices) >= num_devices:
#             visible_devices = free_devices[:num_devices]
#         else:
#             raise RuntimeError('[ERROR] Not enough free CUDA devices available.')
#     devices_str = ','.join(visible_devices)
#     os.environ['CUDA_VISIBLE_DEVICES'] = devices_str
#     return devices_str
#
#
# if __name__ == '__main__':
#     print(find_free_cuda_devices())
