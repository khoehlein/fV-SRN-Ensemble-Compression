import os
import time
import json
import subprocess
import fcntl

from itertools import chain

from utils.misc import get_timestamp_string


LAUNCHER_QUEUE = '/home/hoehlein/PycharmProjects/queue'

RETURN_KEY = 'return'
PID_KEY = 'pid'
TIMESTAMP_KEY = 'timestamp'
DURATION_KEY = 'duration'

INTERPRETER_KEY = 'interpreter'
SCRIPT_KEY = 'script'
CWD_KEY = 'cwd'
POSARGS_KEY = 'posargs'
KWARGS_KEY = 'kwargs'
FLAGS_KEY = 'flags'

OPEN_FOLDER = 'open'
RUNNING_FOLDER = 'running'
DONE_FOLDER = 'done'
FAILED_FOLDER = 'failed'
LOG_FOLDER = 'log'

QUEUE_LOCK_FILE = 'queue.lock'


class PythonProcess(object):
    def __init__(self, interpreter_path, script_path, cwd=None, posargs=None, kwargs=None, flags=None):
        assert os.path.exists(interpreter_path)
        self.interpreter_path = interpreter_path
        assert os.path.exists(script_path)
        self.script_path = script_path
        if cwd is not None:
            assert os.path.isdir(cwd)
        self.cwd = cwd
        if posargs is not None:
            assert type(posargs) == list
        self.posargs = posargs
        if kwargs is not None:
            assert type(kwargs) == dict
        self.kwargs = kwargs
        if flags is not None:
            assert type(flags) == list
        self.flags = flags

    @classmethod
    def from_config_file(cls, config_file_path, kw_mapping=None, kw_prefix='', flag_mapping=None, flag_prefix=''):
        assert os.path.exists(config_file_path) and config_file_path.endswith('.json')
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        interpreter_path = config[INTERPRETER_KEY]
        script_path = config[SCRIPT_KEY]
        cwd = config[CWD_KEY] if CWD_KEY in config else None
        pos_args = config[POSARGS_KEY] if POSARGS_KEY in config else None
        kw_args = config[KWARGS_KEY] if KWARGS_KEY in config else None
        if kw_args is not None:
            if kw_mapping is not None:
                kw_key_set = set(kw_args.keys())
                mapped_keys = kw_key_set.intersection(set(kw_mapping.keys()))
                mapped_kw_args = {
                    **{kw_mapping[key]: kw_args[key] for key in mapped_keys},
                    **{key: kw_args[key] for key in (kw_key_set - mapped_keys)}
                }
                kw_args = mapped_kw_args
            extended_kw_args = {kw_prefix + key: kw_args[key] for key in kw_args}
            kw_args = extended_kw_args
        flags = config[FLAGS_KEY] if POSARGS_KEY in config else None
        if flags is not None:
            if flag_mapping is not None:
                flag_set = set(flags)
                mapped_flag_set = flag_set.intersection(set(flag_mapping.keys()))
                mapped_flags = {flag_mapping[key] for key in mapped_flag_set}.union(flag_set - mapped_flag_set)
                flags = list(mapped_flags)
            flags = [flag_prefix + flag for flag in flags]
        return cls(interpreter_path, script_path, cwd=cwd, posargs=pos_args, kwargs=kw_args, flags=flags)

    def execution_command(self):
        pos_args_string = ['{}'.format(arg) for arg in self.posargs]
        kw_args_string = ['{}'.format(arg) for arg in chain.from_iterable(self.kwargs.items())]
        flags_string = ['{}'.format(arg) for arg in self.flags]
        return [self.interpreter_path, self.script_path] + pos_args_string + kw_args_string + flags_string

    def run(self):
        timestamp = get_timestamp_string()
        time_start = time.time()
        # process = subprocess.Popen(self.execution_command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process = subprocess.Popen(self.execution_command(), stderr=subprocess.PIPE, cwd=self.cwd)
        time_end = time.time()
        duration = (time_end - time_start)
        # outputs, errors = process.communicate()
        # return outputs, errors, process.pid, timestamp, duration
        _, errors = process.communicate()
        log = {
            RETURN_KEY: process.returncode,
            'pid': process.pid,
            'errors': errors.decode(),
            'timestamp': timestamp,
            'duration': duration,
        }
        return process, log


class ConfigLauncher(object):
    def __init__(self, base_path, make_directories=False):
        self.base_path = os.path.abspath(base_path)
        self._pid = os.getpid()
        self._timestamp = get_timestamp_string()
        self._setup_config_directory(make_directories)

    def _setup_config_directory(self, make_directories):
        if not os.path.isdir(self.base_path):
            if make_directories:
                os.makedirs(self.base_path)
            else:
                raise Exception(self._directory_error_message(self.base_path))
        for f in [OPEN_FOLDER, RUNNING_FOLDER, DONE_FOLDER, FAILED_FOLDER]:
            abs_f = os.path.join(self.base_path, f)
            if not os.path.isdir(abs_f):
                if make_directories:
                    os.makedirs(abs_f)
                else:
                    raise Exception(self._directory_error_message(abs_f))
        for f in [DONE_FOLDER, FAILED_FOLDER]:
            log_folder = os.path.join(self.base_path, f, LOG_FOLDER)
            if not os.path.isdir(log_folder):
                if make_directories:
                    os.makedirs(log_folder)
                else:
                    raise Exception(self._directory_error_message(log_folder))
        if not os.path.exists(os.path.join(self.base_path, QUEUE_LOCK_FILE)):
            f = open(os.path.join(self.base_path, QUEUE_LOCK_FILE), 'a')
            f.close()

    @staticmethod
    def _directory_error_message(path):
        return '[ERROR] {} is not a valid directory path'.format(path)

    def run(self):
        while True:
            config_file_name = self._get_open_config_file()
            config_file_lock = self._lock_config_file(RUNNING_FOLDER, config_file_name)
            print('[INFO] Launcher {}: Processing configuration {}'.format(self._pid, config_file_name))
            config_file_path = os.path.join(self.base_path, RUNNING_FOLDER, config_file_name)
            task_log = self._launch_configured_task(config_file_path)
            final_folder = DONE_FOLDER if task_log[RETURN_KEY] == 0 else FAILED_FOLDER
            self._unlock_config_file(config_file_lock)
            final_config_file_name = self._move_config_file(config_file_name, RUNNING_FOLDER, final_folder)
            self._store_log_file(task_log, final_config_file_name, final_folder)

    def _get_open_config_file(self):
        config_file_name = None
        while config_file_name is None:
            queue_lock = self._acquire_queue_lock()
            print('[INFO] Launcher {}: Reading queue'.format(self._pid))
            queue = self.list_configs(OPEN_FOLDER)
            if len(queue) > 0:
                config_file_name = queue[0]
                config_file_name = self._move_config_file(config_file_name, OPEN_FOLDER, RUNNING_FOLDER)
            self._release_queue_lock(queue_lock)
            if config_file_name is None:
                print('[INFO] Launcher {}: No configurations remaining. Waiting for new files...'.format(self._pid))
                time.sleep(10)
        return config_file_name

    def _acquire_queue_lock(self):
        print('[INFO] Launcher {}: Waiting for free queue'.format(self._pid))
        lock_file_path = os.path.join(self.base_path, QUEUE_LOCK_FILE)
        while True:
            try:
                print('[INFO] Launcher {}: Requesting queue access'.format(self._pid))
                lock_file = open(lock_file_path, 'r')
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print('[INFO] Launcher {}: Locking queue'.format(self._pid))
                return lock_file
            except IOError as ex:
                # print('[INFO] Launcher {}: Caught error message: {}'.format(self._pid, ex))
                time.sleep(1)

    def list_configs(self, folder):
        abs_path = os.path.join(self.base_path, folder)
        configs = [
            f for f in os.listdir(abs_path)
            if self._config_file_check(os.path.join(abs_path, f))
        ]
        return configs

    @staticmethod
    def _config_file_check(path):
        if os.path.isfile(path):
            file_name = os.path.basename(path)
            if file_name.startswith('config') and file_name.endswith('.json'):
                return True
        return False

    def _release_queue_lock(self, lock_file):
        print('[INFO] Launcher {}: Releasing queue lock'.format(self._pid))
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

    def _move_config_file(self, file_name, old_location, new_location):
        current_path = os.path.join(self.base_path, old_location, file_name)
        assert os.path.exists(current_path)
        new_path = os.path.join(self.base_path, new_location, file_name)
        if os.path.exists(new_path):
            old_file_name, suffix = file_name.split('.')
            file_name = '{}_{}.{}'.format(old_file_name, get_timestamp_string(), suffix)
            print("[INFO] Launcher {}: New file name {}".format(self._pid, file_name))
            new_path = os.path.join(self.base_path, new_location, file_name)
        os.rename(current_path, new_path)
        return file_name

    def _lock_config_file(self, folder, config_file_name):
        print('[INFO] Launcher {}: Locking config file {}'.format(self._pid, config_file_name))
        file = open(os.path.join(self.base_path, folder, config_file_name), 'r')
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return file

    def _unlock_config_file(self, file):
        print('[INFO] Launcher {}: Releasing config file {}'.format(self._pid, file.name))
        fcntl.flock(file, fcntl.LOCK_UN)
        file.close()

    def _launch_configured_task(self, config_file_path):
        task = PythonProcess.from_config_file(config_file_path)
        _, log = task.run()
        print(
            "[INFO] Launcher {}: File {} finished after {} minutes.".format(
                self._pid, config_file_path, log[DURATION_KEY] // 60
            )
        )
        return log

    def _store_log_file(self, log, final_config_file, final_folder):
        log_file_name = '.'.join(final_config_file.split('.')[:-1] + ['log', 'json'])
        with open(os.path.join(self.base_path, final_folder, LOG_FOLDER, log_file_name), 'w') as log_file:
            json.dump(log, log_file, indent=4, sort_keys=True)

    def queue_directory(self):
        return os.path.join(self.base_path, OPEN_FOLDER)

    def clean_running_folder(self):
        queue_lock = self._acquire_queue_lock()
        running_configs = self.list_configs(RUNNING_FOLDER)
        for cf in running_configs:
            try:
                f = open(os.path.join(self.base_path, RUNNING_FOLDER, cf), 'r')
            except IOError:
                continue
            else:
                f.close()
                self._move_config_file(cf, RUNNING_FOLDER, FAILED_FOLDER)
        self._release_queue_lock(queue_lock)

    def restore_failed_configs(self):
        queue_lock = self._acquire_queue_lock()
        failed_configs = self.list_configs(FAILED_FOLDER)
        for cf in failed_configs:
            self._move_config_file(cf, FAILED_FOLDER, OPEN_FOLDER)
        self._release_queue_lock(queue_lock)

    def reset_folders(self):
        queue_lock = self._acquire_queue_lock()
        for folder in [OPEN_FOLDER, RUNNING_FOLDER, FAILED_FOLDER, DONE_FOLDER]:
            configs = self.list_configs(folder)
            for cf in configs:
                os.remove(os.path.join(self.base_path, folder, cf))
        for folder in [DONE_FOLDER, FAILED_FOLDER]:
            log_folder = os.path.join(self.base_path, folder, LOG_FOLDER)
            logs = os.listdir(log_folder)
            for l in logs:
                os.remove(os.path.join(log_folder, l))
        self._release_queue_lock(queue_lock)

