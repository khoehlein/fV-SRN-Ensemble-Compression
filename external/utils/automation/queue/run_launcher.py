import argparse
from utils.automation.queue import QueueManager, JobLauncher, LAUNCHER_QUEUE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--makedirs', dest='makedirs',
        help='enable ConfigLauncher to create missing directories',
        action='store_true'
    )
    parser.set_defaults(makedirs=False)
    parser.add_argument('--basepath', type=str, help='base path to use for ConfigLauncher', default='')
    args = parser.parse_args()

    base_path = args.basepath if len(args.basepath) > 0 else LAUNCHER_QUEUE
    queue = QueueManager(base_path, make_directories=args.makedirs)
    launcher = JobLauncher(queue)
    launcher.run()
