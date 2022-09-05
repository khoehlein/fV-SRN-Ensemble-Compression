import torch
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CyclicLR


def set_optimizer_options(optim_opts):
    optim_opts.add_option(
        'type', str, choices=['sgd', 'adam', 'adamw', 'rmsprop'], default='sgd',
        help='optimizer to use for experiments'
    )
    optim_opts.add_option(
        'lr', type=float, default=1.e-5,
        help='intial learning rate to use during optimization'
    )
    optim_opts.add_option(
        'weight_decay', type=float, default=1.e-4,
        help='decay rate for weight amplitude during optimization',
    )
    sgd_opts = optim_opts.add_category('sgd')
    sgd_opts.add_option(
        'momentum', type=float, default=0.0,
        help='momentum to use during optimization'
    )
    sgd_opts.add_flag(
        'nesterov', default=False,
        help='use nesterov momentum for SGD optimizer (no effect for other optimizers)',
    )
    adam_opts = optim_opts.add_category('adam')
    adam_opts.add_option(
        'beta_1', type=float, default=0.9,
        help='beta1 parameter for Adam-type optimizers (no effect for SGD)',
    )
    adam_opts.add_option(
        'beta_2', type=float, default=0.999,
        help='beta2 parameter for Adam-type optimizers (no effect for SGD)',
    )
    rms_opts = optim_opts.add_category('rmsprop')
    rms_opts.add_option(
        'momentum', type=float, default=0.0,
        help='momentum for RMSprop optimizer'
    )
    rms_opts.add_option(
        'alpha', type=float, default=0.99,
        help='alpha parameter for RMSprop optimizer'
    )
    rms_opts.add_flag(
        'center', default=False,
        help='use center option for RMSprop optimizer'
    )
    return optim_opts


def build_optimizer(model, opts):
    params = model.parameters()
    if opts['type'] == 'sgd':
        optimizer = SGD(
            params,
            momentum=opts['sgd']['momentum'],
            lr=opts['lr'], weight_decay=opts['weight_decay']
        )
    elif opts['type'] == 'adam':
        optimizer = Adam(
            params,
            lr=opts['lr'],
            betas=(opts['adam']['beta_1'], opts['adam']['beta_2']),
            weight_decay=opts['weight_decay']
        )
    elif opts['type'] == 'adamw':
        optimizer = AdamW(
            params,
            betas=(opts['adam']['beta_1'], opts['adam']['beta_2']),
            lr=opts['lr'], weight_decay=opts['weight_decay']
        )
    elif opts['type'] == 'rmsprop':
        optimizer = RMSprop(
            params,
            lr = opts['lr'], weight_decay=opts['weight_decay'],
            momentum=opts['rmsprop']['momentum'], alpha=opts['rmsprop']['alpha']
        )
    else:
        raise NotImplementedError()
    return optimizer


def set_scheduler_options(sched_opts):
    sched_opts.add_option(
        'type', type=str, choices=['step', 'plateau', 'cyclic', 'none'], default='none',
        help='scheduler to use for experiments',
    )
    sched_opts.add_option(
        'lr_decay', type=float, default=0.1,
        help='gamma parameter for step and plateau schedulers',
    )
    sched_opts.add_option(
        'patience', type=int, default=5,
        help='patience parameter for step and plateau schedulers',
    )
    cyc_opts = sched_opts.add_category('cyclic')
    cyc_opts.add_option(
        'max_lr', type=float, default=1.e-4,
        help='maximum learning rate for cyclic lr scheduling'
    )
    cyc_opts.add_option(
        'periods', type=int, help='number of period cycles per epoch', default=1,
    )
    cyc_opts.add_option(
        'mode', type=str, choices=['triangular', 'triangular2', 'exp_range'], default='triangular'
    )
    cyc_opts.add_option(
        'decay', type=float, help='exponential decay rate when used with mode "exp_range"', default=1.0
    )
    return sched_opts


def build_scheduler(optimizer, opts, **kwargs):
    if opts['type'] == 'none':
        scheduler = None
    elif opts['type'] == 'step':
        scheduler = StepLR(optimizer, step_size=opts['patience'], gamma=opts['lr_decay'])
    elif opts['type'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=opts['lr_decay'], patience=opts['patience'])
    elif opts['type'] == 'cyclic':
        cyc_opts = opts['cyclic']
        assert 'epoch_length' in kwargs
        ssu = (kwargs['epoch_length'] / cyc_opts['periods']) // 2
        ssd = (kwargs['epoch_length'] - cyc_opts['periods'] * ssu) // cyc_opts['periods']
        scheduler = CyclicLR(
            optimizer, base_lr=opts['lr'], max_lr=cyc_opts['max_lr'],
            mode=cyc_opts['mode'], gamma=cyc_opts['decay'],
            step_size_up=ssu, step_size_down=ssd
        )
    else:
        raise NotImplementedError()
    return scheduler


def optimizer_to_device(optim, device):
    # from https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/2
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
