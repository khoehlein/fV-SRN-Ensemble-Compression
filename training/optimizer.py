"""
Optimizer settings for scene representation networks
"""

import torch
import argparse
import json


class Optimizer:
    """
    Class to create and manage the optimizer
    """

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        parser_group = parser.add_argument_group("Optimization")
        parser_group.add_argument('-o', '--optimizer', default='Adam', type=str,
                                  help="The optimizer class, 'torch.optim.XXX'")
        parser_group.add_argument('--optimizer:lr', default=0.01, type=float, help="The learning rate")
        parser_group.add_argument('-i', '--epochs', default=50, type=int,
                                  help="The number of iterations in the training")
        parser_group.add_argument('--optimizer:hyper-params', default="{}", type=str,
                                  help="Additional optimizer parameters parsed as json")
        parser_group.add_argument('--optimizer:scheduler:gamma', type=float, default=0.5,
                                  help='Learning rate decay rate for scheduler')
        parser_group.add_argument('--optimizer:scheduler:step-lr:step-size', type=int, default=500,
                                  help='The learning rate decays every lrStep-epochs (this parameter) by lrGamma factor')
        parser_group.add_argument('--optimizer:scheduler:plateau:patience', type=int, default=10,
                                  help='Patience for plateau scheduler')
        parser_group.add_argument('--optimizer:scheduler:mode', type=str, default='step-lr', choices=['step-lr', 'plateau'],
                                  help="""mode for scheduler (Default: step-lr)""")
        parser_group.add_argument('--optimizer:gradient-clipping:max-norm', type=float, default=None,
                                  help="""max norm for gradient clipping""")

    def __init__(self, opt: dict, parameters, dtype, device):
        self._opt = opt
        self._optimizer_class = getattr(torch.optim, opt['optimizer'])
        self._optimizer_parameters = json.loads(opt['optimizer:hyper_params'])
        self._optimizer_parameters['lr'] = opt['optimizer:lr']
        self._optimizer = self._optimizer_class(parameters, **self._optimizer_parameters)
        self._scheduler_parameters = {
            'mode': opt['optimizer:scheduler:mode'],
            'gamma': opt['optimizer:scheduler:gamma'],
            'hyper_params': {}
        }
        self._build_scheduler(opt=opt)
        self._num_epochs = opt['epochs']
        self._clipping_parameters ={
            'threshold' : opt['optimizer:gradient_clipping:max_norm']
        }

    def _build_scheduler(self, opt=None):
        mode = self._scheduler_parameters['mode']
        gamma = self._scheduler_parameters['gamma']
        if mode == 'step-lr':
            if opt is not None:
                self._scheduler_parameters['hyper_params']['step_size'] = opt['optimizer:scheduler:step_lr:step_size']
            self._scheduler = torch.optim.lr_scheduler.StepLR(
                self._optimizer, **self._scheduler_parameters['hyper_params'],
                gamma=gamma
            )
            self._scheduler_requires_loss = False
        elif mode == 'plateau':
            if opt is not None:
                self._scheduler_parameters['hyper_params']['patience'] = opt['optimizer:scheduler:plateau:patience']
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer, **self._scheduler_parameters['hyper_params'],
                factor=gamma
            )
            self._scheduler_requires_loss = True
        else:
            raise NotImplementedError()

    def reset(self, parameters):
        """
        Resets the optimizer and LR-scheduler
        """
        self._optimizer = self._optimizer_class(parameters, **self._optimizer_parameters)
        self._build_scheduler()

    def num_epochs(self):
        return self._num_epochs

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self, closure):
        self._optimizer.step(closure)

    def post_epoch(self, val_loss: float):
        if self._scheduler_requires_loss:
           self._scheduler.step(val_loss)
        else:
            self._scheduler.step()

    def get_lr(self):
        return [pg['lr'] for pg in self._optimizer.param_groups]

    def clip_grads(self):
        threshold = self._clipping_parameters['threshold']
        if threshold is not None:
            for pg in self._optimizer.param_groups:
                torch.nn.utils.clip_grad_norm(pg['params'], threshold)