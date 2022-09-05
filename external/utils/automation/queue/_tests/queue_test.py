import os
from utils.automation.queue import QueueManager, JobMaker, JobLauncher

if __name__ == '__main__':
    base_path = '/home/hoehlein/PycharmProjects/deployment/delllat94/weatherbench_deep_features/utils/experiments/automation/_tests/'
    queue = QueueManager(os.path.join(base_path, 'test_queue'), make_directories=True)
    interpreter = '/home/hoehlein/anaconda3/bin/python'
    maker1 = JobMaker(interpreter, os.path.join(base_path, 'fail_job.py'))
    maker2 = JobMaker(interpreter, os.path.join(base_path, 'finish_job.py'))
    maker1.export_jobs(queue)
    maker2.export_jobs(queue)
    launcher = JobLauncher(queue)
    launcher.run()
    print('Finished')