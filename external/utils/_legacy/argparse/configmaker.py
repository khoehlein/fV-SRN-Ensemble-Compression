from utils._legacy.configmaker import CONFIG_FILE_FLAG


def add_config_maker_support(parser):
    parser.add_option(CONFIG_FILE_FLAG, type=str, help='config file used for launching the experiment', default='none')