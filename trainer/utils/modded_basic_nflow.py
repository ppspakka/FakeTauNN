import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np

from nflows import distributions, flows, transforms, utils
import nflows.nn.nets as nn_

from pathlib import Path

from nflows.transforms.base import Transform
from nflows.transforms.autoregressive import AutoregressiveTransform
from nflows.transforms import made as made_module

from modded_spline import (
    unconstrained_rational_quadratic_spline,
    rational_quadratic_spline,
)
import modded_spline
from nflows.utils import torchutils

# from nflows.transforms import splines
from torch.nn.functional import softplus

from modded_coupling import PiecewiseCouplingTransformRQS, CouplingTransformMAF
from modded_base_flow import FlowM


class AffineCouplingTransform(CouplingTransformMAF):
    """An affine coupling layer that scales and shifts part of the variables.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.

    The user should supply `scale_activation`, the final activation function in the neural network producing the scale tensor.
    Two options are predefined in the class.
    `DEFAULT_SCALE_ACTIVATION` preserves backwards compatibility but only produces scales <= 1.001.
    `GENERAL_SCALE_ACTIVATION` produces scales <= 3, which is more useful in general applications.
    """

    DEFAULT_SCALE_ACTIVATION = lambda x: torch.sigmoid(x + 2) + 1e-3
    GENERAL_SCALE_ACTIVATION = lambda x: (softplus(x) + 1e-3).clamp(0, 30)

    def __init__(
        self,
        mask,
        transform_net_create_fn,
        unconditional_transform=None,
        scale_activation=GENERAL_SCALE_ACTIVATION,
        init_identity=True,
    ):
        self.scale_activation = scale_activation
        self.init_identity = init_identity
        super().__init__(mask, transform_net_create_fn, unconditional_transform)

    def _transform_dim_multiplier(self):
        return 2

    def _scale_and_shift(self, transform_params):
        unconstrained_scale = transform_params[:, self.num_transform_features :, ...]
        shift = transform_params[:, : self.num_transform_features, ...]
        if self.init_identity:
            shift = shift - 0.5414
        scale = self.scale_activation(unconstrained_scale)
        return scale, shift

    def _coupling_transform_forward(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = inputs * scale + shift
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _coupling_transform_inverse(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet


class MLP(nn.Module):
    """A standard multi-layer perceptron."""

    def __init__(
        self,
        in_shape,
        out_shape,
        context_features,
        hidden_sizes,
        activation=F.relu,
        activate_output=False,
        batch_norm=False,
        dropout_probability=0.0,
    ):
        """
        Args:
            in_shape: tuple, list or torch.Size, the shape of the input.
            out_shape: tuple, list or torch.Size, the shape of the output.
            hidden_sizes: iterable of ints, the hidden-layer sizes.
            activation: callable, the activation function.
            activate_output: bool, whether to apply the activation to the output.
        """
        super().__init__()
        self._in_shape = in_shape
        self._out_shape = out_shape
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._activate_output = activate_output
        self._batch_norm = batch_norm
        self.dropout = nn.Dropout(p=dropout_probability)

        if len(hidden_sizes) == 0:
            raise ValueError("List of hidden sizes can't be empty.")

        if context_features is not None:
            self.initial_layer = nn.Linear(in_shape + context_features, hidden_sizes[0])
        else:
            self.initial_layer = nn.Linear(in_shape, hidden_sizes[0])

        if self._batch_norm:
            self.layer_norm_layers = nn.ModuleList(
                [torch.nn.LayerNorm(sizes) for sizes in hidden_sizes]
            )
        self._hidden_layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:])
            ]
        )
        self.final_layer = nn.Linear(hidden_sizes[-1], np.prod(out_shape))

    def forward(self, inputs, context=None):
        # if inputs.shape[1:] != self._in_shape:
        #     raise ValueError(
        #         "Expected inputs of shape {}, got {}.".format(
        #             self._in_shape, inputs.shape[1:]
        #         )
        #     )

        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        outputs = temps
        outputs = self._activation(outputs)

        for i, hidden_layer in enumerate(self._hidden_layers):
            outputs = hidden_layer(outputs)
            # NOTE batch norm is broken right now
            if self._batch_norm:
                outputs = self.layer_norm_layers[i](outputs)
            outputs = self._activation(outputs)
            outputs = self.dropout(outputs)

        outputs = self.final_layer(outputs)
        if self._activate_output:
            outputs = self._activation(outputs)
        # outputs = outputs.reshape(-1, *torch.Size(self._out_shape))
        print(outputs.shape)

        return outputs


class EmbedATT(nn.Module):
    """Use attention on embedding of inputs"""

    def __init__(
        self,
        in_shape,
        embed_shape,
        out_shape,
        context_features,
        hidden_sizes,
        activation=F.relu,
        activate_output=False,
        layer_norm=False,
        dropout_probability=0.0,
        num_heads=5,
    ):
        """
        Args:
            in_shape: tuple, list or torch.Size, the shape of the input.
            out_shape: tuple, list or torch.Size, the shape of the output.
            hidden_sizes: iterable of ints, the hidden-layer sizes.
            activation: callable, the activation function.
            activate_output: bool, whether to apply the activation to the output.
        """
        super().__init__()
        self._in_shape = in_shape
        self._embed_shape = embed_shape
        self._out_shape = out_shape
        self._context_features = context_features
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._activate_output = activate_output
        self._layer_norm = layer_norm
        self.dropout = nn.Dropout(p=dropout_probability)
        self.num_heads = num_heads

        if len(hidden_sizes) == 0:
            raise ValueError("List of hidden sizes can't be empty.")

        if context_features is not None:
            input = in_shape + context_features
            self.embedding = nn.Linear(input, input * embed_shape)
        else:
            input = in_shape
            self.embedding = nn.Linear(input, input * embed_shape)

        if self._layer_norm:
            self.layer_norm_layers = nn.ModuleList(
                [torch.nn.LayerNorm(sizes) for sizes in hidden_sizes]
            )

        self.attention = torch.nn.MultiheadAttention(
            embed_dim=self._embed_shape, num_heads=self.num_heads, batch_first=True
        )

        self._hidden_layers = nn.ModuleList(
            [
                nn.Linear(
                    self._embed_shape * (self._in_shape + self._context_features), 128
                ),
                nn.Linear(128, 128),
                nn.Linear(128, 128),
            ]
        )
        self.final_layer = nn.Linear(128, np.prod(out_shape))

    def forward(self, inputs, context=None):
        # if inputs.shape[1:] != self._in_shape:
        #     raise ValueError(
        #         "Expected inputs of shape {}, got {}.".format(
        #             self._in_shape, inputs.shape[1:]
        #         )
        #     )

        if context is None:
            temps = self.embedding(inputs)
            temps = temps.view(-1, self._in_shape, self._embed_shape)
        else:
            temps = self.embedding(torch.cat((inputs, context), dim=1))
            temps = temps.view(
                -1, self._in_shape + self._context_features, self._embed_shape
            )
        outputs = temps
        # attention
        outputs, _ = self.attention(outputs, outputs, outputs, need_weights=False)
        outputs = outputs.reshape((len(inputs), -1))

        for i, hidden_layer in enumerate(self._hidden_layers):
            outputs = hidden_layer(outputs)
            # NOTE batch norm is broken right now
            if self._layer_norm:
                outputs = self.layer_norm_layers[i](outputs)
            outputs = self._activation(outputs)
            outputs = self.dropout(outputs)

        outputs = self.final_layer(outputs)
        if self._activate_output:
            outputs = self._activation(outputs)
        # outputs = outputs.reshape(-1, *torch.Size(self._out_shape))

        return outputs


class MaskedAffineAutoregressiveTransformM(AutoregressiveTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        init_identity=True,
        affine_type="sigmoid",
    ):
        self.features = features
        made = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        self._epsilon = 5e-2
        self.init_identity = init_identity
        self.affine_type = affine_type
        if init_identity:
            torch.nn.init.constant_(made.final_layer.weight, 0.0)
            if self.affine_type == "softplus":
                torch.nn.init.constant_(
                    made.final_layer.bias,
                    0.5414,  # the value k to get softplus(k) = 1.0
                )
            elif self.affine_type == "sigmoid":
                torch.nn.init.constant_(
                    made.final_layer.bias,
                    -7.906,  # the value k to get sigmoid(k+1) = 1.0
                )
            elif self.affine_type == "atan":
                torch.nn.init.constant_(
                    made.final_layer.bias, 1  # the value k to get atan(k) = 1.0
                )

        super(MaskedAffineAutoregressiveTransformM, self).__init__(made)

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        shift = shift.clamp(-50, 50)
        if self.affine_type == "sigmoid":
            scale = 1000 * torch.sigmoid(unconstrained_scale + 1.0) + self._epsilon
        elif self.affine_type == "softplus":
            scale = ((F.softplus(unconstrained_scale)) + self._epsilon).clamp(0, 1)
        elif self.affine_type == "atan":
            scale = (1000 * torch.atan(unconstrained_scale / 1000)).clamp(0.001, 50)

        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
    
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        if self.affine_type == "sigmoid":
            scale = 1000 * torch.sigmoid(unconstrained_scale + 1.0) + self._epsilon
        elif self.affine_type == "softplus":
            scale = ((F.softplus(unconstrained_scale)) + self._epsilon).clamp(0, 1)
        elif self.affine_type == "atan":
            scale = (1000 * torch.atan(unconstrained_scale / 1000)).clamp(0.001, 50)
        log_scale = torch.log(scale)
        # print(scale, shift)
        outputs = (inputs - shift) / scale
        
        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, autoregressive_params):
        # split_idx = autoregressive_params.size(1) // 2
        # unconstrained_scale = autoregressive_params[..., :split_idx]
        # shift = autoregressive_params[..., split_idx:]
        # return unconstrained_scale, shift
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        unconstrained_scale = autoregressive_params[..., 0]
        shift = autoregressive_params[..., 1]
        if self.init_identity:
            if self.affine_type == "sigmoid":
                shift = shift + 7.906
            elif self.affine_type == "softplus":
                shift = shift - 0.5414
            elif self.affine_type == "atan":
                shift = shift - 1
        # print(unconstrained_scale, shift)
        return unconstrained_scale, shift


class MaskedPiecewiseRationalQuadraticAutoregressiveTransformM(AutoregressiveTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        init_identity=True,
        min_bin_width=modded_spline.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=modded_spline.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=modded_spline.DEFAULT_MIN_DERIVATIVE,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        autoregressive_net = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        if init_identity:
            torch.nn.init.constant_(autoregressive_net.final_layer.weight, 0.0)
            torch.nn.init.constant_(
                autoregressive_net.final_layer.bias,
                np.log(np.exp(1 - min_derivative) - 1),
            )

        super().__init__(autoregressive_net)

    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs,
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)


class PiecewiseRationalQuadraticCouplingTransformM(PiecewiseCouplingTransformRQS):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        apply_unconditional_transform=False,
        img_shape=None,
        init_identity=True,
        min_bin_width=modded_spline.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=modded_spline.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=modded_spline.DEFAULT_MIN_DERIVATIVE,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = lambda features: PiecewiseRationalQuadraticCDF(
                shape=[features] + (img_shape if img_shape else []),
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            )
        else:
            unconditional_transform = None

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        else:
            return self.num_bins * 3 + 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn(
                "Inputs to the softmax are not scaled down: initialization might be bad."
            )

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs,
        )


def create_random_transform(param_dim):
    """Create the composite linear transform PLU.
    Arguments:
        input_dim {int} -- dimension of the space
    Returns:
        Transform -- nde.Transform object
    """

    return transforms.CompositeTransform(
        [
            transforms.RandomPermutation(features=param_dim),
            transforms.LULinear(param_dim, identity_init=True),
        ]
    )


def create_mixture_flow_model(input_dim, context_dim, base_kwargs):
    """Build NSF (neural spline flow) model. This uses the nsf module
    available at https://github.com/bayesiains/nsf.
    This models the posterior distribution p(x|y).
    The model consists of
        * a base distribution (StandardNormal, dim(x))
        * a sequence of transforms, each conditioned on y
    Arguments:
        input_dim {int} -- dimensionality of x
        context_dim {int} -- dimensionality of y
        num_flow_steps {int} -- number of sequential transforms
        base_transform_kwargs {dict} -- hyperparameters for transform steps: should put num_transform_blocks=10,
                          activation='elu',
                          batch_norm=True
    Returns:
        Flow -- the model
    """

    distribution = distributions.StandardNormal((input_dim,))
    transform = []
    for _ in range(base_kwargs["num_steps_maf"]):
        transform.append(
            MaskedAffineAutoregressiveTransformM(
                features=input_dim,
                use_residual_blocks=base_kwargs["use_residual_blocks_maf"],
                num_blocks=base_kwargs["num_transform_blocks_maf"],
                hidden_features=base_kwargs["hidden_dim_maf"],
                context_features=context_dim,
                dropout_probability=base_kwargs["dropout_probability_maf"],
                use_batch_norm=base_kwargs["batch_norm_maf"],
                init_identity=base_kwargs["init_identity"],
                affine_type=base_kwargs["affine_type"],
            )
        )
        if base_kwargs["permute_type"] != "no-permutation":
            transform.append(create_random_transform(param_dim=input_dim))

    for _ in range(base_kwargs["num_steps_arqs"]):
        transform.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransformM(
                features=input_dim,
                tails="linear",
                use_residual_blocks=base_kwargs["use_residual_blocks_arqs"],
                hidden_features=base_kwargs["hidden_dim_arqs"],
                num_blocks=base_kwargs["num_transform_blocks_arqs"],
                tail_bound=base_kwargs["tail_bound_arqs"],
                num_bins=base_kwargs["num_bins_arqs"],
                context_features=context_dim,
                dropout_probability=base_kwargs["dropout_probability_arqs"],
                use_batch_norm=base_kwargs["batch_norm_arqs"],
                init_identity=base_kwargs["init_identity"],
            )
        )
        if base_kwargs["permute_type"] != "no-permutation":
            transform.append(create_random_transform(param_dim=input_dim))

    for i in range(base_kwargs["num_steps_caf"]):
        if base_kwargs["coupling_net"] == "mlp":
            transform.append(
                AffineCouplingTransform(
                    mask=utils.create_alternating_binary_mask(
                        features=input_dim, even=(i % 2 == 0)
                    ),
                    transform_net_create_fn=(
                        lambda in_features, out_features: MLP(
                            in_shape=in_features,
                            out_shape=out_features,
                            hidden_sizes=base_kwargs[
                                "hidden_dim_caf"
                            ],  # list of hidden layer dimensions
                            context_features=context_dim,
                            activation=F.relu,
                            activate_output=False,
                            batch_norm=base_kwargs["batch_norm_caf"],
                            dropout_probability=base_kwargs["dropout_probability_caf"],
                        )
                    ),
                )
            )
        elif base_kwargs["coupling_net"] == "att":
            transform.append(
                AffineCouplingTransform(
                    mask=utils.create_alternating_binary_mask(
                        features=input_dim, even=(i % 2 == 0)
                    ),
                    transform_net_create_fn=(
                        lambda in_features, out_features: EmbedATT(
                            in_shape=in_features,
                            embed_shape=base_kwargs["att_embed_shape"],
                            out_shape=out_features,
                            hidden_sizes=base_kwargs[
                                "hidden_dim_caf"
                            ],  # list of hidden layer dimensions
                            context_features=context_dim,
                            activation=F.relu,
                            activate_output=False,
                            layer_norm=base_kwargs["batch_norm_caf"],
                            dropout_probability=base_kwargs["dropout_probability_caf"],
                            num_heads=base_kwargs["att_num_heads"],
                        )
                    ),
                )
            )
        if base_kwargs["permute_type"] != "no-permutation":
            transform.append(create_random_transform(param_dim=input_dim))

    transform_fnal = transforms.CompositeTransform(transform)

    flow = FlowM(transform_fnal, distribution)

    # Store hyperparameters. This is for reconstructing model when loading from
    # saved file.

    flow.model_hyperparams = {
        "input_dim": input_dim,
        "context_dim": context_dim,
        "base_kwargs": base_kwargs,
    }

    return flow


def save_model(
    epoch,
    model,
    scheduler,
    train_history,
    test_history,
    name,
    model_dir=None,
    optimizer=None,
):
    """Save a model and optimizer to file.
    Args:
        model:      model to be saved
        optimizer:  optimizer to be saved
        epoch:      current epoch number
        model_dir:  directory to save the model in
        filename:   filename for saved model
    """

    if model_dir is None:
        raise NameError("Model directory must be specified.")

    filename = name + f"_@epoch_{epoch}.pt"
    resume_filename = "checkpoint-latest.pt"

    p = Path(model_dir)
    p.mkdir(parents=True, exist_ok=True)

    dict = {
        "train_history": train_history,
        "test_history": test_history,
        "model_hyperparams": model.model_hyperparams,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        dict["scheduler_state_dict"] = scheduler.state_dict()
        dict["last_lr"] = scheduler.get_last_lr()

    torch.save(dict, p / filename)
    torch.save(dict, p / resume_filename)


def load_mixture_model(device, model_dir=None, filename=None):
    """Load a saved model.
    Args:
        filename:       File name
    """

    if model_dir is None:
        raise NameError(
            "Model directory must be specified."
            " Store in attribute PosteriorModel.model_dir"
        )

    p = Path(model_dir)
    checkpoint = torch.load(p / filename, map_location="cpu")

    model_hyperparams = checkpoint["model_hyperparams"]
    # added because of a bug in the old create_mixture_flow_model function
    try:
        if checkpoint["model_hyperparams"]["base_transform_kwargs"] is not None:
            checkpoint["model_hyperparams"]["base_kwargs"] = checkpoint[
                "model_hyperparams"
            ]["base_transform_kwargs"]
            del checkpoint["model_hyperparams"]["base_transform_kwargs"]
    except KeyError:
        pass
    train_history = checkpoint["train_history"]
    test_history = checkpoint["test_history"]

    # Load model
    model = create_mixture_flow_model(**model_hyperparams)
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.to(device)

    # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model.eval()

    # Load optimizer
    scheduler_present_in_checkpoint = "scheduler_state_dict" in checkpoint.keys()

    # If the optimizer has more than 1 param_group, then we built it with
    # flow_lr different from lr
    if len(checkpoint["optimizer_state_dict"]["param_groups"]) > 1:
        flow_lr = checkpoint["last_lr"]
    elif checkpoint["last_lr"] is not None:
        flow_lr = checkpoint["last_lr"][0]
    else:
        flow_lr = None

    # Set the epoch to the correct value. This is needed to resume
    # training.
    epoch = checkpoint["epoch"]
    optim_state = checkpoint["optimizer_state_dict"]
    return (
        model,
        scheduler_present_in_checkpoint,
        flow_lr,
        epoch,
        train_history,
        test_history,
        optim_state,
    )
