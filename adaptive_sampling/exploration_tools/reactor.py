#!/usr/bin/env python3
import sys, os
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

from ..interface.sampling_data import MDInterface
from ..units import *

class Reactor(ABC):
    """Abstract class for molecular dynamics based reaction space exploration using reactor methods

    Args:
        md: Object of the MD Interface
        init_temp: initial temperature of MD
        equil_temp: equilibrium temperature of MD
        verbose: print verbose information
        output_freq: frequency in steps for writing outputs
    """
    def __init__(
        self,
        r_min: float,
        r_max: float,
        md: MDInterface,
        verbose: bool = True,
        #mols_array = [],
        *args,
        **kwargs,
    ):
        self.the_md = md
        self.verbose = verbose
        self.bond_order_list = []
        self.r_max = r_max/BOHR_to_ANGSTROM
        self.r_min = r_min/BOHR_to_ANGSTROM
     
        
    @abstractmethod
    def step_bias(self):
        pass




