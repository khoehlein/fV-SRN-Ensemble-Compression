import argparse
import os
import socket
from enum import Enum


class DebugMode(Enum):
    DEBUG = 'debug'
    PRODUCTION = 'production'


DEBUG_MODE: DebugMode = None


def set_debug_mode(args):
    global DEBUG_MODE
    DEBUG_MODE = DebugMode.DEBUG if args['mode'] == 'debug' else DebugMode.PRODUCTION
    print(f'[INFO] Running scripts with path configurations for {DEBUG_MODE.value} mode.')
    return DEBUG_MODE


PROJECT_BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUT_BASE_PATH = '/path/to/output/dir'
INTERPRETER_PATH = '/path/to/bin/python'
DATA_BASE_PATH = {'host-name': '/path/to/data'}


def get_project_base_path():
    return PROJECT_BASE_PATH


def get_data_base_path():
    host_name = socket.gethostname()
    return DATA_BASE_PATH[host_name]


def get_output_base_path():
    return OUTPUT_BASE_PATH


OUTPUT_DIR_NAME = 'runs'
LOG_DIR_NAME = 'log'


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=[DebugMode.DEBUG.value, DebugMode.PRODUCTION.value], default=DebugMode.DEBUG.value)
    return parser


def get_output_directory(experiment_name, overwrite=False, return_output_dir=False, return_log_dir=False):
    experiment_dir = os.path.join(OUTPUT_BASE_PATH, experiment_name)
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)
    else:
        if not overwrite:
            raise RuntimeError(f'[ERROR] Experiment directory {experiment_dir} exists already.')
    out = [experiment_dir]
    if return_output_dir:
        output_dir = os.path.join(experiment_dir, OUTPUT_DIR_NAME)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        out.append(output_dir)
    if return_log_dir:
        log_dir = os.path.join(experiment_dir, LOG_DIR_NAME)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        out.append(log_dir)
    return tuple(out)
