import argparse
from collections import namedtuple, OrderedDict


Argument = namedtuple('Argument', ['id', 'type', 'help', 'default', 'choices'])

Flag = namedtuple('Flag', ['id', 'help', 'default'])


class ScriptOptions(object):

    def __init__(self, id: str):
        self.id = id
        self._categories = OrderedDict({})
        self._arguments = OrderedDict({})

    def _check_id(self, id):
        assert id not in self._categories, \
            '[ERROR] Argument IDs must be unique. Template {} already has a category {}'.format(self.id, id)
        assert id not in self._arguments, \
            '[ERROR] Argument IDs must be unique. Template {} already has an argument {}'.format(self.id, id)

    def add_category(self, id=None, options=None):
        if options is None:
            assert id is not None
            options = ScriptOptions(id)
        else:
            if id is None:
                id = options.id
        self._check_id(id)
        self._categories[id] = options
        return options

    def add_option(self, id, type, help=None, default=None, choices=None):
        self._check_id(id)
        self._arguments[id] = Argument(id, type, help, default, choices)

    def add_flag(self, id, help=None, default=None):
        self._check_id(id)
        assert default is not None, '[ERROR] Flags must have a default setting'
        self._arguments[id] = Flag(id, help, default)

    def write_to_parser(self, parser, prefix=None):
        for id, arg in self._arguments.items():
            if type(arg) == Argument:
                self._write_argument_to_parser(arg, parser, prefix)
            elif type(arg) == Flag:
                self._write_flag_to_parser(arg, parser, prefix)
        for id, c in self._categories.items():
            c.write_to_parser(parser, prefix=(f'{id}_' if prefix is None else f'{prefix}{id}_'))
        return parser

    def get_parser(self):
        parser = argparse.ArgumentParser()
        return self.write_to_parser(parser)

    def _write_argument_to_parser(self, arg, parser, prefix):
        if prefix is None:
            prefix = ''
        arg_dest = prefix + arg.id
        props = dict(
            type=arg.type, choices=arg.choices,
            help=arg.help, default=arg.default,
        )
        if arg.default is not None:
            arg_name = '--' + arg_dest.replace('_', '-')
            parser.add_argument(arg_name, dest=arg_dest, **props)
        else:
            parser.add_argument(arg_dest, **props)

    def _write_flag_to_parser(self, arg, parser, prefix):
        if prefix is None:
            prefix = ''
        arg_dest = prefix + arg.id
        arg_name = '--' + arg_dest.replace('_', '-')
        no_arg_name = '--' + (prefix + 'no-' + arg.id).replace('_', '-')
        parser.add_argument(
            arg_name, dest=arg_dest,
            help=arg.help,
            action='store_true'
        )
        parser.add_argument(
            no_arg_name, dest=arg_dest,
            help='do not {}'.format(arg.help) if arg.help is not None else None,
            action='store_false'
        )
        parser.set_defaults(**{arg_name: arg.default})

    def read_settings(self, args, prefix=None):
        summary = {}
        if prefix is None:
            prefix = ''
        for id in self._arguments:
            arg_dest = prefix + id
            summary[id] = getattr(args, arg_dest)
        for id, c in self._categories.items():
            summary[id] = c.read_settings(args, prefix=(f'{id}_' if prefix is None else f'{prefix}{id}_'))
        return summary

    @staticmethod
    def flatten_settings(settings, prefix=''):
        out ={}
        for key, params in settings.items():
            if type(params) == dict:
                out.update(ScriptOptions.flatten_settings(params, prefix=f'{prefix}{key}_'))
            else:
                out[f'{prefix}{key}'] = params
        return out







