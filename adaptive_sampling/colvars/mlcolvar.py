import torch
import numpy as np
from ..units import *

from .utils import convert_coordinate_system


class MLCOLVAR:
    """Calculate CV from trained PyTorch model

    Args:
        model: filename of the PyTorch model (*.pt, *.ptc)
        coordinate_system: specify input coordinate system for model features ('cartesian', 'cv_space', 'zmatrix')
        cv_def: definition of input features for PyTorch model
        component: if model has multiple output nodes, specify index that should be used as CV
        unit_conversion_factor: conversion factor for input coordinates to convert Bohr to units of training data (nm, angstrom, ...)
        requires_grad: if gradient should be computed
        device: torch device ('CPU', 'CUDA', ...)
        ndim: space dimensions (set to 2 for 2D test potentials)
    """

    def __init__(
        self,
        model: str = None,
        coordinate_system: str = "cv_space",
        cv_def: dict = None,
        component: int = None,
        unit_conversion_factor: float = 1.0,
        device: str = "CPU",
        ndim: int = 3,
    ):
        self.model = torch.load(model, weights_only=False)
        self.coordinate_system = coordinate_system
        self.cv_def = cv_def
        self.component = component
        self.unit_conversion_factor = unit_conversion_factor
        self.device = device
        self.ndim = ndim
        self.cv = None
        self.gradient = None

    def calc(self, coords: np.array, **kwargs):
        """Calc CV from ML model and its gradient using autograd

        Args:
            coords: cartesian coordinates
        """
        # calculate feature space
        coords = torch.mul(coords, self.unit_conversion_factor)
        feature_space = convert_coordinate_system(
            coords,
            self.cv_def,
            coord_system=self.coordinate_system,
            ndim=self.ndim,
        )

        # get CV from PyTorch model
        self.cv = self.model(feature_space, **kwargs)
        if self.component is not None:
            self.cv = self.cv[self.component]

        if coords.requires_grad:
            self.gradient = torch.autograd.grad(self.cv, coords, allow_unused=True)[0]
            self.gradient = self.gradient.detach().numpy()

        return self.cv
