import argparse
from typing import Dict, Any, Optional, Union

from torch import Tensor

from inference.model.output_parameterization.density_output import CHOICES as DENSITY_OUTPUTS_BY_NAME
from inference.model.output_parameterization.rgbo_output import CHOICES as RGBO_OUTPUTS_BY_NAME

from data.datasets import OutputMode
from data.datasets.output_mode import MultivariateOutputMode
from inference.model.output_parameterization import IOutputParameterization, BackendOutputMode
from inference.model.output_parameterization.multivariate_output import MultivariateClampedOutput
from inference.model.scene_representation_network.evaluation_mode import EvaluationMode

_choices_by_output_mode = {
    OutputMode.DENSITY: DENSITY_OUTPUTS_BY_NAME,
    OutputMode.RGBO: RGBO_OUTPUTS_BY_NAME,
    OutputMode.MULTIVARIATE: {} # currently not supported
}


class PyrendererOutputParameterization(IOutputParameterization):

    OUTPUT_MODE: OutputMode = None # will be set during parser initialization

    @classmethod
    def set_output_mode(cls, output_mode: OutputMode):
        # assert output_mode != OutputMode.MULTIVARIATE, '[ERROR] Multivariate output is currently not supported in OutputParameterization'
        cls.OUTPUT_MODE = output_mode

    @classmethod
    def init_parser(cls, parser: argparse.ArgumentParser, output_mode: Optional[OutputMode] = None):
        group = parser.add_argument_group('OutputParameterization')
        prefix = '--network:output:'
        if cls.OUTPUT_MODE is None:
            cls.set_output_mode(output_mode)
        if output_mode is None:
            output_mode = cls.OUTPUT_MODE
        assert output_mode is not None, '[ERROR] Output mode must be given either through class argument or keyword argument'
        choices = list(_choices_by_output_mode[output_mode].keys())
        group.add_argument(
            prefix + 'parameterization-method', choices=choices, type=str, default='',
            help="""
            The possible outputs of the network:
            - density: a scalar density is produced that is then mapped to color via the TF.
                * soft-clamp: Sigmoid clamp to [0, 1]
                * hard-clamp: Hard clamp to [0, 1]
                * mixed: Clamp hard for rendering
                * direct: noop during training, clamp to [0,1] during rendering
            - rgbo: the network directly estimates red, green, blue, opacity/absorption. The TF is fixed during training and inference.                      
                * soft-clamp: Sigmoid clamp to [0, 1] for color, softplus clamping to [0, infty] for absorption
                * direct: noop for training, clamp to [0,1] for color, [0, infty] for absorption for rendering
                * exp: Sigmoid clamp to [0, 1] for color, exponential clamping to [0, infty] for absorption
            """
        )

    @classmethod
    def from_dict(cls, args: Dict[str, Any], output_mode: Optional[Union[OutputMode, MultivariateOutputMode]]=None):
        if output_mode is None:
            assert cls.OUTPUT_MODE is not None, '[ERROR] OutputParameterization output mode must be set before instance can be created from arguments.'
            output_mode = cls.OUTPUT_MODE
        if isinstance(output_mode, OutputMode):
            parameterization_mode = args['network:output:parameterization_method']
            parameterization_class = _choices_by_output_mode[output_mode][parameterization_mode]
            return cls(parameterization_class())
        elif isinstance(output_mode, MultivariateOutputMode):
            return cls(MultivariateClampedOutput(output_mode.num_channels))
        else:
            raise NotImplementedError()

    def __init__(self, parameterization: IOutputParameterization):
        super(PyrendererOutputParameterization, self).__init__()
        self._parameterization = parameterization

    def input_channels(self) -> int:
        return self._parameterization.input_channels()

    def output_channels(self) -> int:
        return self._parameterization.output_channels()

    def output_mode(self) -> OutputMode:
        return self._parameterization.output_mode()

    def backend_output_mode(self) -> BackendOutputMode:
        return self._parameterization.backend_output_mode()

    def forward(self, network_output: Tensor, evaluation_mode: EvaluationMode) -> Tensor:
        return self._parameterization.forward(network_output, evaluation_mode)
