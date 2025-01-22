import torch
import numpy as np
from ..units import *

from .utils import convert_coordinate_system
import torch.nn as nn 

class MLCOLVAR():
    """Calculate CV from trained PyTorch model

    Args:
        model: filename of the torchscript model (*.pt, *.ptc)
        coordinate_system: specify input coordinate system for model features ('cartesian', 'cv_space', 'zmatrix')
        cv_def: definition of input features for PyTorch model
        cv_idx: if model has multiple output nodes, specify index of output feature that should be used as CV
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
        cv_idx: int = None,
        unit_conversion_factor: float = None,
        device: str = None,
        ndim: int = 3,
    ):
        # load and optimize torchscript model
        self.device = device
        self._model = torch.jit.load(model, self.device)
        self._model.eval()
        try:        
            self._model = torch.jit.optimize_for_inference(self._model)
        except:
            print(" >>> WARNING: optimization for inference failed. Please use torch version to >= 1.10")
            self._model = torch.jit.freeze(self._model)

        # other parameters
        self.coordinate_system = coordinate_system
        self.cv_def = cv_def
        self.cv_idx = cv_idx
        self.unit_conversion_factor = unit_conversion_factor if not None else 1.0
        self.ndim = ndim
        self.cv = None
        self.gradient = None

    def forward(self, coords: torch.tensor):
        """Obtain the mlcolvar from forward pass through model

        Args:
            coords: cartesian coordinates
        """
        if self.device is not None:
            coords = coords.to(self.device, non_blocking=True)
        coords_new = coords * self.unit_conversion_factor
            
        # calculate feature space
        feature_space = convert_coordinate_system(
            coords_new,
            self.cv_def,
            coord_system=self.coordinate_system,
            ndim=self.ndim,
        )

        # get CV from PyTorch model
        cv = self._model(feature_space)
        self.cv = cv[self.cv_idx] if self.cv_idx is not None else cv

        return self.cv
