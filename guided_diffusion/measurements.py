"""Measurement model used by the staged MIROC6 distilled-CG workflow."""

import torch
import torch.nn.functional as F


_OPERATORS = {}
_NOISES = {}


def register_operator(name):
    def decorator(cls):
        _OPERATORS[name] = cls
        return cls
    return decorator


def get_operator(name, **kwargs):
    try:
        return _OPERATORS[name](**kwargs)
    except KeyError as error:
        raise ValueError(f"Unsupported operator: {name}") from error


def register_noise(name):
    def decorator(cls):
        _NOISES[name] = cls
        return cls
    return decorator


def get_noise(name, **kwargs):
    try:
        noise = _NOISES[name](**kwargs)
    except KeyError as error:
        raise ValueError(f"Unsupported noise model: {name}") from error
    noise.__name__ = name
    return noise


@register_operator('bilinear_interp_cycle_to32x64to720x1440')
class BilinearInterpolationCycle:
    """Apply A = upsample_720x1440(downsample_32x64(x)).

    The inference code represents both the observation and A(x) on the
    720x1440 grid.  As in the original experiment, ``transpose`` uses the
    same cycle as an approximate adjoint for the CG correction.
    """

    def __init__(self, in_shape, device=None):
        if tuple(in_shape[-2:]) != (720, 1440):
            raise ValueError(f"Expected a 720x1440 field, got {tuple(in_shape[-2:])}.")
        self.low_shape = (32, 64)
        self.high_shape = (720, 1440)
        self.device = device

    def forward(self, data, **kwargs):
        low_resolution = F.interpolate(
            data, size=self.low_shape, mode='bilinear', align_corners=False
        )
        return F.interpolate(
            low_resolution, size=self.high_shape, mode='bilinear', align_corners=False
        )

    def transpose(self, data, **kwargs):
        return self.forward(data, **kwargs)


@register_noise('gaussian')
class GaussianNoise:
    def __init__(self, sigma):
        self.sigma = float(sigma)

    def __call__(self, data):
        return data + self.sigma * torch.randn_like(data)

