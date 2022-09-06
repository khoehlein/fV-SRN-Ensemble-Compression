import os

PROJECT_BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_BASE_PATH = '~/PycharmProjects/results/fvsrn'


def get_project_base_path():
    return PROJECT_BASE_PATH


def get_output_base_path():
    return OUTPUT_BASE_PATH