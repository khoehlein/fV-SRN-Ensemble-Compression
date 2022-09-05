import torch.nn as nn
from datetime import datetime
from inspect import getmro


class VersionedModule(nn.Module):

    __version__ = '1.0'

    def __init__(self):
        super(VersionedModule, self).__init__()
        self._version_codes = dict()
        for cls in getmro(self.__class__):
            name = cls.__name__
            module = cls.__module__
            if module is not None:
                name = '.'.join([module, name])
            if not '__version__' in cls.__dict__:
                raise Exception(
                    '[ERROR] {} as subclass of VersionedModule must override class attribute __version__.'.format(name)
                )
            self._version_codes.update({
                name: cls.__version__
            })
            if cls is VersionedModule:
                break

    def get_code_version(self):
        return self._version_codes
