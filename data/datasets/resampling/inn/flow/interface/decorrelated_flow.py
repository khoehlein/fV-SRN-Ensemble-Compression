from typing import Any, Tuple

import torch
from torch import nn, Tensor

from .flow_direction import FlowDirection
from .invertible_transform import InvertibleTransform
from .conditional_flow import ConditionalFlow
from ..log_likelihood import LogLikelihoodPerDimension
from ..latent_space import GaussianLatentSpace


class Decorrelator(nn.Module):

    def __init__(
            self,
            flow: ConditionalFlow,
            latent_space: GaussianLatentSpace
    ):
        super(Decorrelator, self).__init__()
        self.flow = flow
        self.latent_space = latent_space
        self.log_likelihood_tracker = LogLikelihoodPerDimension(dimensions=latent_space.dimensions)
        self.flow.add_determinant_tracker(self.log_likelihood_tracker)

    def forward(self, *data):
        self.log_likelihood_tracker.activate().reset()
        conditions = torch.cat(data, dim=1)
        # This module assumes that all provided data and condition fields have the same spatial shape!
        z = self.latent_space.generate_sample(batch_size=data[0].shape[0])
        device = data[0].device
        z = [zc.to(device=device) for zc in z]
        prior_ll_per_dim = self.latent_space.compute_log_likelihood(*z)
        self.log_likelihood_tracker.add_prior_log_likelihood(
            prior_ll_per_dim=prior_ll_per_dim, dimensions=self.latent_space.dimensions
        )
        u = self.flow.forward(*z, conditions, direction=FlowDirection.REVERSE)
        ll = self.log_likelihood_tracker.get_value()
        self.log_likelihood_tracker.deactivate()
        return u, ll


class DecorrelatedFlow(InvertibleTransform):

    def __init__(self, flow: InvertibleTransform, decorrelator: Decorrelator):
        super(DecorrelatedFlow, self).__init__({
            FlowDirection.FORWARD: flow.get_number_of_flow_ports(FlowDirection.FORWARD),
            FlowDirection.REVERSE: flow.get_number_of_flow_ports(FlowDirection.REVERSE),
        }, num_condition_ports=flow.get_number_of_condition_ports())
        self.flow = flow
        self.decorrelator = decorrelator

    def forward_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        noise, noise_ll = self.decorrelator(*data)
        flow_data, condition_data = self.split_inputs(*data, direction=FlowDirection.FORWARD)
        flow_data = [d + n for d, n in zip(flow_data, noise)]
        if self.has_determinant_tracker():
            # negative noise likelihood for reweighting !
            self.update_determinant_trackers(- noise_ll, self.decorrelator.latent_space.dimensions)
        return self.flow.forward(*flow_data, *condition_data, direction=FlowDirection.FORWARD)

    def reverse_transform(self, *data: Any) -> Tuple[Any, ...]:
        return self.flow.forward(*data, direction=FlowDirection.REVERSE)

    def add_determinant_tracker(self, ll_per_dim: LogLikelihoodPerDimension):
        self.flow.add_determinant_tracker(ll_per_dim)
        return super(DecorrelatedFlow, self).add_determinant_tracker(ll_per_dim)
