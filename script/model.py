"""Shared model construction utilities for prior training, distillation, and inference."""

import torch
import torch.nn as nn
import yaml
from diffusers import UNet2DModel


def load_yaml(file_path: str) -> dict:
    with open(file_path) as file:
        return yaml.safe_load(file)


class DiffusersUNetWrapper(nn.Module):
    """Concatenate five physical fields with the eight fixed condition fields."""

    def __init__(self, model_config, device):
        super().__init__()
        self.unet = UNet2DModel.from_config(model_config)
        self.in_channels = model_config['in_channels']
        self.out_channels = model_config['out_channels']
        self.condition_channels = self.in_channels - self.out_channels
        if self.condition_channels <= 0:
            raise ValueError('The model must reserve channels for fixed conditions.')
        self.to(device)

    def forward(self, x, timesteps, fixed_condition):
        if x.shape[1] != self.out_channels:
            raise ValueError(f'Expected {self.out_channels} physical channels, got {x.shape[1]}.')
        if fixed_condition is None or fixed_condition.shape[1] != self.condition_channels:
            raise ValueError(f'Expected {self.condition_channels} fixed-condition channels.')
        return self.unet(torch.cat([x, fixed_condition], dim=1), timesteps.long()).sample
