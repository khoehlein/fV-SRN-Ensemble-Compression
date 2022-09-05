"""
Helpers to evaluate the scene representation networks in world- and screen space.
Input: Network and the current batch from the dataloader
"""

from typing import Optional, Union

import numpy as np
import torch
from torch import nn
import pyrenderer

from common.raytracing import Raytracing
from inference.losses.lossnet import LossNetScreen, LossNetWorld
from inference.model import ISceneRepresentationNetwork
from inference.model.pyrenderer import PyrendererSRN


def _get_output_mode(network: Union[ISceneRepresentationNetwork, PyrendererSRN]) -> str:
    if isinstance(network, ISceneRepresentationNetwork):
        return network.output_mode().split(':')[0]
    elif isinstance(network, PyrendererSRN):
        return network.backend_output_mode().value.split(':')[0]
    else:
        raise NotImplementedError()


class EvaluateScreen:
    def __init__(self, network: Union[ISceneRepresentationNetwork, PyrendererSRN], evaluator:pyrenderer.IImageEvaluator,
                 loss: Optional[LossNetScreen],
                 image_width:int, image_height:int, train:bool,
                 disable_inversion_trick: bool,
                 dtype, device):

        self._network = network
        self._use_checkpointing = False if disable_inversion_trick else train
        self._loss = loss

        self._network_output_mode = _get_output_mode(network)  # trim options
        if self._network_output_mode == 'density' and train:
            raise ValueError( "For now, only rgbo-output is supported in screen-space for networks in training mode, no training through a TF yet")
        self._raytracing = Raytracing(evaluator, self._network_output_mode, 1.0, image_width, image_height, dtype, device)

    def __call__(self, dataloader_batch):
        """
        Evaluates the network.
        If loss is None, the returns total_loss and partial_losses is also None
        :param dataloader_batch: the tuple of pytorch-cuda tensors from the screen-space dataloader
        :param loss: the screen-space loss network
        :return: image (B,4,H,W), totol_loss, partial_losses
        """

        camera, target, tf_index, time_index, ensemble_index, stepsize = dataloader_batch
        if isinstance(stepsize, torch.Tensor):
            stepsize = stepsize[0].item()
        self._raytracing.set_stepsize(stepsize)
        if self._use_checkpointing:
            image = self._raytracing.checkpointed_trace(self._network, camera, network_args=[tf_index, time_index, ensemble_index, 'screen'])
        else:
            image = self._raytracing.full_trace_forward(self._network, camera, network_args=[tf_index, time_index, ensemble_index, 'screen'])

        if self._loss is None:
            total_loss = None
            partial_losses = None
        else:
            total_loss, partial_losses = self._loss(image, target, return_individual_losses=True)

        return image, total_loss, partial_losses

    def loss_names(self):
        return self._loss.loss_names()


class EvaluateMultivariateScreen(EvaluateScreen):

    class MultivariateNetworkWrapperForRendering(nn.Module):

        def __init__(self, network, active_output):
            super(EvaluateMultivariateScreen.MultivariateNetworkWrapperForRendering, self).__init__()
            self.network = network
            self.active_output = active_output

        def use_direction(self):
            return self.network.use_direction()

        def forward(self, *args):
            prediction = self.network.forward(*args)
            return prediction[:, [self.active_output]]

    def __call__(self, dataloader_batch):
        """
        Evaluates the network.
        If loss is None, the returns total_loss and partial_losses is also None
        :param dataloader_batch: the tuple of pytorch-cuda tensors from the screen-space dataloader
        :param loss: the screen-space loss network
        :return: image (B,4,H,W), totol_loss, partial_losses
        """

        camera, targets, tf_index, time_index, ensemble_index, stepsize = dataloader_batch
        images = []
        total_loss = None
        partial_losses = None
        for i, target in enumerate(targets):
            if isinstance(stepsize, torch.Tensor):
                stepsize = stepsize[0].item()
            self._raytracing.set_stepsize(stepsize)
            self._network.output_parameterization._parameterization.active_output = i
            if self._use_checkpointing:
                image = self._raytracing.checkpointed_trace(self._network, camera, network_args=[tf_index, time_index, ensemble_index, 'screen'])
            else:
                image = self._raytracing.full_trace_forward(self._network, camera, network_args=[tf_index, time_index, ensemble_index, 'screen'])

            if self._loss is not None:
                new_total_loss, new_partial_losses = self._loss(image, target, return_individual_losses=True)
                if total_loss is None:
                    total_loss = new_total_loss
                else:
                    total_loss = total_loss + new_total_loss
                new_partial_losses = {key + f'_c{i}': new_partial_losses[key] for key in new_partial_losses}
                if partial_losses is None:
                    partial_losses = new_partial_losses
                else:
                    partial_losses.update(new_partial_losses)

            images.append(image)

        return images, total_loss, partial_losses


class EvaluateWorld:
    def __init__(self, network: Union[ISceneRepresentationNetwork, PyrendererSRN], evaluator:pyrenderer.IImageEvaluator,
                 loss: Optional[LossNetWorld], dtype, device):
        self._network = network
        self._loss = loss

        self._network_output_mode = _get_output_mode(network) # trim options
        self._loss_input_mode = self._network_output_mode if loss is None else loss.mode()

        assert self._network_output_mode in ['density', 'rgbo']
        assert self._loss_input_mode in ['density', 'rgbo']

        if self._network_output_mode == 'density' and self._loss_input_mode=='rgbo':
            raise NotImplementedError("Training through the TF is not supported yet")
        if self._network_output_mode == 'rgbo' and self._loss_input_mode=='density':
            raise ValueError("The loss function expects densities, but the network already predictions derived colors")

        if network.use_direction():
            raise ValueError("The network requires directions, but this is not available in world-space evaluation")

    def __call__(self, dataloader_batch):
        """
        Evaluates the current batch
        :param dataloader_batch: the batch from the world-space data loader
        :return: values (B,C), total_loss, partial_losses
        """

        position, target, tf, time, ensemble = dataloader_batch
        # evaluate network
        predictions = self._network(position, tf, time, ensemble, 'world')
        # loss
        if self._loss is None:
            total_loss = None
            partial_losses = None
        else:
            total_loss, partial_losses = self._loss(predictions, target, return_individual_losses=True)

        return predictions, total_loss, partial_losses

    def loss_names(self):
        return self._loss.loss_names()


class EvaluateOrderedWorld(EvaluateWorld):

    def __call__(self, dataloader_batch):
        predictions = []
        total_loss = None
        partial_losses = None
        num_sub_batches = len(dataloader_batch)

        for i, sub_batch in enumerate(dataloader_batch):
            sub_batch_predictions, sub_batch_loss, sub_batch_partial_losses = super(EvaluateOrderedWorld, self).__call__(sub_batch)
            predictions.append(sub_batch_predictions)
            if total_loss is None:
                total_loss = sub_batch_loss
            else:
                total_loss = total_loss + sub_batch_loss
            if partial_losses is None:
                partial_losses = sub_batch_partial_losses
            else:
                partial_losses = {
                    key: partial_losses[key] + sub_batch_partial_losses[key]
                    for key in sub_batch_partial_losses
                }

        partial_losses = {
            key: partial_losses[key] / num_sub_batches
            for key in partial_losses
        }
        total_loss = total_loss / num_sub_batches
        return predictions, total_loss, partial_losses


class EvaluateWorldAndRegularization(EvaluateWorld):

    def __call__(self, dataloader_batch):
        predictions, total_loss, partial_losses = super(EvaluateWorldAndRegularization, self).__call__(dataloader_batch)
        try:
            reg_loss = self._network.latent_features.volumetric_features.compute_regularization()
        except Exception as err:
            partial_losses.update({
                'volumetric_features:regularization': 0.,
                'volumetric_features:num_active_channels': np.nan,
            })
        else:
            total_loss = total_loss + reg_loss
            num_active_channels = self._network.latent_features.volumetric_features.num_active_channels()
            partial_losses.update({
                'volumetric_features:regularization': reg_loss.item(),
                'volumetric_features:num_active_channels': num_active_channels,
            })
        try:
            reg_loss = self._network.latent_features.ensemble_features.compute_regularization()
        except Exception as err:
            partial_losses.update({
                'ensemble_features:regularization': 0,
                'ensemble_features:num_active_channels': np.nan,
            })
        else:
            total_loss = total_loss + reg_loss
            num_active_channels = self._network.latent_features.ensemble_features.num_active_channels()
            partial_losses.update({
                'ensemble_features:regularization': reg_loss.item(),
                'ensemble_features:num_active_channels': num_active_channels,
            })
        return predictions, total_loss, partial_losses

    def loss_names(self):
        return [l for l in self._loss.loss_names()] + [
            'ensemble_features:regularization', 'ensemble_features:num_active_channels',
            'volumetric_features:regularization', 'volumetric_features:num_active_channels']