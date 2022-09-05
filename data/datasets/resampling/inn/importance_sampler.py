import argparse
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
from torch.optim import Adam

from inference import IFieldEvaluator
from data.datasets.resampling import IImportanceSampler, CoordinateBox
from .flow.interface import NormalizingFlow, FlowDirection
from .flow.log_likelihood import LogLikelihoodPerDimension
from .flow.transforms.quadratic_spline_flow import RationalQuadraticSplineFlow
from .flow.transforms.random_permute import RandomPermute
from data.datasets.sampling import ISampler, RandomUniformSampler


def get_warp_flow(dimension: int, num_transforms: int, hidden_channels: int, spline_order: int):

    def get_processor():
        return nn.Sequential(
            nn.Linear(dimension // 2, hidden_channels),
            nn.PReLU(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.PReLU(hidden_channels),
            nn.Linear(hidden_channels, (dimension - dimension // 2) * (2 * spline_order + 1))
        )
    transforms = []
    for _ in range(num_transforms - 1):
        transforms += [
            RationalQuadraticSplineFlow(3, get_processor(), swap_groups=False),
            RandomPermute(3)
        ]
    transforms.append(RationalQuadraticSplineFlow(3, get_processor(), swap_groups=False))
    flow = NormalizingFlow.chain(*transforms, direction=FlowDirection.FORWARD)
    return flow


class WarpingNetworkImportanceSampler(IImportanceSampler):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('WarpingNetworkImportanceSampler')
        prefix = '--importance-sampler:warp-net:'
        group.add_argument(prefix + 'num-transforms', type=int, default=4, help="""
        number of rational spline transforms for the warping flow
        """)
        group.add_argument(prefix + 'hidden-channels', type=int, default=32, help="""
        number of hidden channels in processing units of the warping flow
        """)
        group.add_argument(prefix + 'spline-order', type=int, default=8, help="""
        number of nodes for spline computations in rational spline blocks
        """)
        group.add_argument(prefix + 'num-batches', type=int, default=8, help="""
        number of training batches per update step
        """)
        group.add_argument(prefix + 'num-samples-per-batch', type=int, default=64**3, help="""
        number of sample positions per training batch
        """)
        group.add_argument(prefix + 'lr', type=float, default=1.e-3, help="""
        learning rate for flow optimizer
        """)
        group.add_argument(prefix + 'weight-decay', type=float, default=1.e-4, help="""
        weight decay rate for flow optimizer
        """)
        group.add_argument(prefix + 'entropy-regularization', type=float, default=1.e-4, help="""
        weight decay rate for flow optimizer
        """)
        group.add_argument(prefix + 'log-p-regularization', type=float, default=1.e-4, help="""
        weight decay rate for flow optimizer
        """)

    @classmethod
    def from_dict(cls, args, dimension=None, sampler: Optional[ISampler] = None, root_box: Optional[CoordinateBox] = None, device=None):
        prefix = 'importance_sampler:warp_net:'
        kws = {
            key: args[prefix + key] for key in [
                'num_transforms', 'hidden_channels', 'spline_order', 'lr', 'weight_decay',
                'num_batches', 'num_samples_per_batch', 'entropy_regularization', 'log_p_regularization'
            ]}

        return cls(**kws, dimension=dimension, sampler=sampler, device=device, root_box=root_box)

    def __init__(
            self,
            dimension=None,
            num_transforms=4, hidden_channels=64, spline_order=8, lr=5.e-4, weight_decay=1.e-4,
            num_batches=8, num_samples_per_batch=64**3, entropy_regularization=5., log_p_regularization=1.,
            sampler: Optional[ISampler] = None, root_box: Optional[CoordinateBox] = None, device=None
    ):
        if dimension is None and sampler is not None:
            dimension = sampler.dimension
        if dimension is None and root_box is not None:
            dimension = root_box.dimension
        assert dimension is not None
        if sampler is not None:
            assert dimension == sampler.dimension
            if device is not None:
                assert sampler.device == device
            device = sampler.device
        else:
            assert device is not None
            sampler = RandomUniformSampler(dimension, device=device)
        super(WarpingNetworkImportanceSampler, self).__init__(dimension, root_box, device)
        self.sampler = sampler
        self.flow = get_warp_flow(dimension, num_transforms, hidden_channels, spline_order).to(device)
        self.ll_per_dim = LogLikelihoodPerDimension(dimension)
        self.flow.add_determinant_tracker(self.ll_per_dim)
        self.optimizer = Adam(self.flow.parameters(), lr=lr, weight_decay=weight_decay)
        self.num_batches = num_batches
        self.num_samples_per_batch = num_samples_per_batch
        self.entropy_regularization = entropy_regularization
        self.log_p_regularization = log_p_regularization

    def generate_samples(self, num_samples: int, evaluator: IFieldEvaluator, **kwargs):
        self._train_flow_on_evaluator(evaluator)
        positions, weights = self._generate_samples(num_samples)
        if self.root_box is not None:
            positions = self.root_box.rescale(positions)
        return positions, weights

    def _train_flow_on_evaluator(self, evaluator: IFieldEvaluator):
        self.flow.train()
        self.ll_per_dim.activate()
        for _ in range(self.num_batches):
            self.optimizer.zero_grad()
            self.ll_per_dim.reset()
            positions = self.sampler.generate_samples(self.num_samples_per_batch)
            with torch.no_grad():
                losses = evaluator.evaluate(positions)[:, 0]
            self.flow.forward(positions, direction=FlowDirection.FORWARD)
            log_p = self.ll_per_dim.flush()  * self.dimension
            p = torch.exp(log_p)
            sqrt_p = torch.exp(log_p / 2.)
            expected_loss = torch.mean(losses.detach() * sqrt_p)
            loss = - expected_loss
            if self.entropy_regularization > 0.:
                entropy = - torch.mean(p * log_p)  # optimize for large average loss
                loss = loss - self.entropy_regularization * entropy
            if self.log_p_regularization > 0.:
                log_p_amp = torch.mean(torch.abs(log_p) ** 2)
                loss = loss + self.log_p_regularization * log_p_amp
            loss.backward()
            self.optimizer.step()
        self.flow.eval()
        self.ll_per_dim.deactivate()

    def _generate_samples(self, num_samples: int):
        with torch.no_grad():
            self.flow.eval()
            positions = self.sampler.generate_samples(num_samples)
            self.ll_per_dim.activate().reset()
            positions = self.flow.forward(positions, direction=FlowDirection.REVERSE)[0]
            log_p = self.ll_per_dim.flush()
            self.ll_per_dim.deactivate()
            weights = torch.exp(log_p)
        return positions, weights

    # def _test_fw_bw(self):
    #     with torch.no_grad():
    #         samples = self.sampler.generate_samples(10)
    #         self.flow.eval()
    #         self.ll_per_dim.activate()
    #         self.ll_per_dim.reset()
    #         out = self.flow.forward(samples, direction=FlowDirection.REVERSE)
    #         p_rev = self.ll_per_dim.flush()
    #         self.ll_per_dim.reset()
    #         reverted = self.flow.forward(*out, direction=FlowDirection.FORWARD)
    #         p_fw = self.ll_per_dim.flush()
    #         self.ll_per_dim.reset()
    #         self.ll_per_dim.deactivate()
    #         print(p_rev - p_fw)


def _test_sampler():

    class Evaluator(IFieldEvaluator):

        def __init__(self, dimension, device=None):
            super(Evaluator, self).__init__(dimension, 1, device)
            self.direction = 4 * torch.tensor([1] * dimension, device=device)[None, :]# torch.randn(1, dimension, device=device)
            self.offset = torch.tensor([0.5] * dimension, device=device)[None, :] # torch.randn(1, dimension, device=device)

        def forward(self, positions: Tensor) -> Tensor:
            return torch.sum(self.direction * (positions - self.offset), dim=-1) ** 2

    device = torch.device('cuda:0')
    evaluator = Evaluator(3, device=device)

    for p in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.]:
        sampler = WarpingNetworkImportanceSampler(dimension=3, device=device, num_batches=20, entropy_regularization=0.,
                                           log_p_regularization=p)
        # sampler._test_fw_bw()
        samples, weights = sampler.generate_samples(1000, evaluator)
        c = evaluator.evaluate(samples)
        samples = samples.data.cpu().numpy()
        c = c[:, 0].data.cpu().numpy()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=c)
        plt.show()
        plt.close()


if __name__ == '__main__':
    _test_sampler()