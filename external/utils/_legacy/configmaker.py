import os
import random
from itertools import product
import uuid
import json
from .configlauncher import SCRIPT_KEY, INTERPRETER_KEY, CWD_KEY, POSARGS_KEY, KWARGS_KEY, FLAGS_KEY
from utils.misc import get_timestamp_string

CONFIG_FILE_FLAG = '--configfile'


class ConfigMaker(object):
    def __init__(
            self,
            interpreter_path, script_path, cwd=None,
            posargs=None, kwargs=None, flags=None,
            randomize=False, max_num_files=None,
            _verify_interpreter=True, _verify_script=True, _verify_cwd=True
    ):
        self.interpreter_path, self.script_path, self.cwd, self.posargs, self.kwargs, self.flags = self._parse_inputs(
            interpreter_path, script_path, cwd,
            posargs, kwargs, flags,
            _verify_interpreter, _verify_script, _verify_cwd
        )
        kwargs_list = [list(product([key], values)) for key, values in self.kwargs.items()]
        all_settings = list(product(*(self.posargs + kwargs_list + self.flags)))
        if randomize:
            all_settings = self._randomize(all_settings, max_num_files)
        elif max_num_files > 0:
            all_settings = all_settings[:min(max_num_files, len(all_settings))]
        else:
            raise Exception()
        num_posargs = len(self.posargs)
        num_kwargs = len(self.kwargs)
        self.configurations = [
            {
                INTERPRETER_KEY: self.interpreter_path,
                SCRIPT_KEY: self.script_path,
                CWD_KEY: self.cwd,
                POSARGS_KEY: setting[:num_posargs],
                KWARGS_KEY: dict(setting[num_posargs:(num_posargs + num_kwargs)]),
                FLAGS_KEY: setting[(num_posargs + num_kwargs):],
            }
            for setting in all_settings
        ]

    @staticmethod
    def _parse_inputs(
            interpreter_path, script_path, cwd,
            posargs, kwargs, flags,
            _verify_interpreter, _verify_script, _verify_cwd
    ):
        if _verify_interpreter:
            assert os.path.exists(interpreter_path)
        interpreter_path = os.path.abspath(interpreter_path)
        if _verify_script:
            assert os.path.exists(script_path) and script_path.endswith('.py')
        script_path = os.path.abspath(script_path)
        if _verify_cwd and cwd is not None:
            assert os.path.isdir(cwd)
        if posargs is None:
            posargs = []
        assert type(posargs) == list
        posargs_parsed = [args if type(args) == list else [args] for args in posargs]
        if kwargs is None:
            kwargs = dict()
        assert type(kwargs) == dict
        kwargs_parsed = {key: args if type(args) == list else [args] for key, args in kwargs.items()}
        if flags is None:
            flags = []
        assert type(flags) == list
        flags_parsed = [flag if type(flag) == list else [flag] for flag in flags]
        return interpreter_path, script_path, cwd, posargs_parsed, kwargs_parsed, flags_parsed

    @staticmethod
    def _randomize(all_settings, max_num_files):
        if max_num_files is None:
            random.shuffle(all_settings)
        elif max_num_files > 0:
            all_settings = random.sample(all_settings, max_num_files)
        else:
            raise Exception()
        return all_settings

    def export_config_files(self, path, make_directories=False, clear_directory=False, confirm_clearing=True):
        if not os.path.isdir(path):
            if make_directories:
                os.makedirs(path)
            else:
                raise Exception('[ERROR] {} is not a valid directory path.'.format(path))
        if clear_directory:
            self._clear_directory(path, confirm_clearing)
        print('[INFO] Writing {} new configuration files'.format(len(self.configurations)))
        for config in self.configurations:
            config_file_name = self._get_config_file_name()
            config[KWARGS_KEY].update({CONFIG_FILE_FLAG: config_file_name})
            with open(os.path.join(path, config_file_name), 'w') as config_file:
                json.dump(config, config_file, indent=4, sort_keys=False)

    @staticmethod
    def _clear_directory(path, confirm_clearing):
        old_config_files = [f for f in os.listdir(path) if (f.startswith('config') and f.endswith('.json'))]
        num_files = len(old_config_files)
        if num_files > 0:
            print('[INFO] Found {} old configuration file{}'.format(num_files, 's' if num_files > 1 else ''))
            if confirm_clearing:
                print('[INFO] Proceed with deleting? (yes/no)')
                user_input = input()
                while user_input not in ['yes', 'no']:
                    print('[INFO] "{}" is not a valid answer. Try again with "yes" or "no".'.format(user_input))
                    user_input = input()
                if user_input == 'yes':
                    print('[INFO] Got answer "yes". Deleting {} old configuration file{}'.format(num_files, 's' if num_files > 1 else ''))
                    for cf in old_config_files:
                        os.remove(os.path.join(path, cf))
                else:
                    print('[INFO] Got answer "no". Aborting deletion.')
            else:
                for cf in old_config_files:
                    os.remove(os.path.join(path, cf))

    @staticmethod
    def _get_config_file_name():
        return 'config_{}_{}.json'.format(get_timestamp_string(), uuid.uuid4().hex)
