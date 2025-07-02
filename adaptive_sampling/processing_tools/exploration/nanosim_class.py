import sys
from typing import final
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import rdkit.Chem.rdmolops as rdmolops
import rdkit.Chem.rdmolfiles as rdmolfiles
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from .utils import (
    merge_neighbor_anions,
    merge_neighbor_radicals,
    is_in,
    round_bond_order_numpy,
    int_atom,
    divide_traj,
    correct_valences,
    is_metal,
    check_electroneg
)
from .read_write_utils import read_pop_file, read_bo_file, read_frag_file, read_traj_file, write_frag_file

# dictionary for standard experimental bond lengths (in Angstrom) defined globally to be used in classes and functions

# change this with data from here: https://cccbdb.nist.gov/diatomicexpbondx.asp
global __std_bond_lengths__
__std_bond_lengths__ = {
    "H":  {"H":   0.741, 
           "Li":  1.595, "Be": 1.343, 
           "B":   1.232, "C":  1.120, "N":  1.036, "O":  0.970, "F":  0.917, "Ne": np.NaN, 
           "Na":  1.887, "Mg": 1.730, "Al": 1.648, "Si": 1.520, "P":  1.422, "S":   1.341, "Cl": 1.275, "Ar": np.NaN,
           "K":   2.243, "Ca": 2.003, "Sc": 1.775, "Ti": 1.785, "V": np.NaN, "Cr":  1.655, "Cu": 1.463, "Zn":  1.595, "Ga": 1.663, "Ge": 1.588, "As": 1.535, "Se": 1.475, "Br": 1.414, 
           "Sb": np.NaN, "Te": 1.656, "I":  1.609},
    "Li": {"H":   1.595, 
           "Li":  2.673, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":   1.688, "F":   1.564, "Ne": np.NaN, 
           "Na":  2.889, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P":  np.NaN, "S":   2.150, "Cl":  2.021, "Ar": np.NaN,
           "K":   3.270, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V":  np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  2.170, 
           "Sb": np.NaN, "Te": np.NaN, "I":   2.392},
    "Be": {"H":   1.343, 
           "Li": np.NaN, "Be":  2.460, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":   1.331, "F":  1.361, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":   1.742, "Cl":  1.797, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br": np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "B":  {"H":   1.232, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":   1.590, "C":   1.491, "N":   1.325, "O":   1.205, "F":  1.267, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":   1.609, "Cl":  1.719, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  1.888, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "C":  {"H":   1.120, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":   1.491, "C":   1.243, "N":   1.172, "O":   1.128, "F":  1.276, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al":  1.955, "Si":  1.722, "P":  1.562, "S":   1.535, "Cl":  1.649, "Ar": np.NaN,
           "K":  np.NaN, "Ca":  2.302, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": 1.676, "Br":  1.821, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "N":  {"H":   1.036, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":   1.325, "C":   1.172, "N":   1.098, "O":   1.154, "F":  1.317, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al":  1.786, "Si":	1.575, "P":	 1.491, "S":   1.497, "Cl":  1.611, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": 1.618, "Se": 1.652, "Br": 1.765, 	 
           "Sb":  1.835, "Te": np.NaN, "I":  np.NaN},           
    "O":  {"H":   0.970, 
           "Li":  1.688, "Be":  1.331, 
           "B":   1.205, "C":   1.128, "N":   1.154, "O":   1.208, "F":  1.354, "Ne": np.NaN, 
           "Na":  2.052, "Mg":  1.749, "Al":  1.618, "Si":  1.510, "P":	 1.476, "S":   1.481, "Cl":  1.596, "Ar": np.NaN,
           "K":  np.NaN, "Ca":  1.822, "Sc":  1.668, "Ti":	1.620, "V":	 1.589, "Cr": np.NaN, "Cu":  1.724, "Zn": np.NaN, "Ga": 1.743, "Ge": 1.625, "As": np.NaN, "Se": 1.639, "Br": 1.718, 	 
           "Sb": np.NaN, "Te":  1.825, "I":   1.868},
    "F":  {"H":   0.917, 
           "Li":  1.564, "Be":  1.361, 
           "B":   1.267, "C":   1.276, "N":   1.317, "O":   1.354, "F":  1.412, "Ne": np.NaN, 
           "Na":  1.926, "Mg":  1.750, "Al":  1.654, "Si": 1.604, "P": 1.593, "S":   1.599, "Cl":  1.628, "Ar": np.NaN,
           "K":   2.171, "Ca":  1.967, "Sc":  1.788, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": 1.745, "Zn": np.NaN, "Ga": 1.774, "Ge": 1.745, "As": 1.736, "Se": np.NaN, "Br": 1.759, 
           "Sb":  1.918, "Te": np.NaN, "I":   1.910},
    "Ne": {"H":  np.NaN, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":   np.NaN, "F": np.NaN, "Ne": 3.100, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":  np.NaN, "Cl": np.NaN, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "Na": {"H":   1.887, 
           "Li":  2.889, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":   2.052, "F": 1.926, "Ne": np.NaN, 
           "Na":  3.079, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":  2.489, "Cl": 2.361, "Ar": np.NaN,
           "K":   3.589, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  2.502, 
           "Sb": np.NaN, "Te": np.NaN, "I":  2.711},
    "Mg": {"H":   1.730, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":   1.749, "F": 1.750, "Ne": np.NaN, 
           "Na": np.NaN, "Mg":  3.891, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":  2.143, "Cl": 2.199, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "Al": {"H":   1.648, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":   1.955, "N":   1.786, "O":   1.618, "F":  1.654, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al":  2.701, "Si": np.NaN, "P":  2.400, "S":   2.029, "Cl":  2.130, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  2.295, 
           "Sb": np.NaN, "Te": np.NaN, "I":  2.537},
    "Si": {"H":   1.520, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":   1.722, "N":   1.575, "O":   1.510, "F":  1.604, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si":  2.246, "P":  2.078, "S":   1.929, "Cl":  2.061, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": 2.058, "Br":  2.209, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "P":  {"H":   1.422, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":   1.562, "N":   1.491, "O":   1.476, "F":  1.593, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al":  2.400, "Si":  2.078, "P":  1.893, "S":   1.900, "Cl":  2.018, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": 2.450, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  2.171, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "S":  {"H":   1.341, 
           "Li":  2.150, "Be":  1.742, 
           "B":   1.609, "C":   1.535, "N":   1.497, "O":   1.481, "F":  1.599, "Ne": np.NaN, 
           "Na":  2.489, "Mg":  2.143, "Al":  2.029, "Si":  1.929, "P":  1.900, "S":   1.889, "Cl":  1.975, "Ar": np.NaN,
           "K":  np.NaN, "Ca":  2.318, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn":  2.046, "Ga": np.NaN, "Ge": 2.012, "As": np.NaN, "Se": 2.037, "Br":  np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "Cl": {"H":   1.275, 
           "Li":  2.021, "Be":  1.797, 
           "B":   1.719, "C":   1.649, "N":   1.611, "O":  1.596, "F":  1.628, "Ne": np.NaN, 
           "Na":  2.361, "Mg":  2.199, "Al":  2.130, "Si": 2.061, "P":  2.018, "S":   1.975, "Cl": 1.988, "Ar": np.NaN,
           "K":   2.667, "Ca":  2.437, "Sc": np.NaN, "Ti": 2.265, "V": np.NaN, "Cr": np.NaN, "Cu": 2.051, "Zn": np.NaN, "Ga": 2.202, "Ge": 2.164, "As": np.NaN, "Se": np.NaN, "Br":  2.136, 
           "Sb":  2.335, "Te": np.NaN, "I":   2.321},
    "Ar": {"H":  np.NaN, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":  np.NaN, "F": np.NaN, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":  np.NaN, "Cl": np.NaN, "Ar":  3.758,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "K":  {"H":   2.243, 
           "Li":  3.270, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":  np.NaN, "F":  2.171, "Ne": np.NaN, 
           "Na":  3.589, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":  np.NaN, "Cl": 2.667, "Ar": np.NaN,
           "K":   3.905, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  2.821, 
           "Sb": np.NaN, "Te": np.NaN, "I":   3.048},
    "Ca": {"H":   2.003, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":   2.302, "N":  np.NaN, "O":   1.822, "F":  1.967, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":  2.318, "Cl": 2.437, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  2.594, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "Sc": {"H":   1.775, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":   1.668, "F":  1.788, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":  np.NaN, "Cl": np.NaN, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "Ti": {"H":   1.785, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":   1.620, "F": np.NaN, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":  np.NaN, "Cl":  2.265, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "V":  {"H":  np.NaN, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":   1.589, "F": np.NaN, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":  np.NaN, "Cl": np.NaN, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "Cr": {"H":   1.655, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":  np.NaN, "F": np.NaN, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":  np.NaN, "Cl": np.NaN, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "Cu": {"H":   1.463, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":   1.724, "F":  1.745, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":  np.NaN, "Cl":  2.051, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu":  2.220, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "Zn": {"H":   1.595, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":  np.NaN, "F": np.NaN, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":   2.046, "Cl": np.NaN, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "Ga": {"H":   1.663, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":   1.743, "F":  1.774, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P":  2.450, "S":  np.NaN, "Cl":  2.202, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As":  2.530, "Se": np.NaN, "Br":  2.352, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "Ge": {"H":   1.588, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":   1.625, "F":  1.745, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":   2.012, "Cl":  2.164, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se":  2.135, "Br": np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "As": {"H":   1.535, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":   1.618, "O":  np.NaN, "F": 1.736, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":  np.NaN, "Cl": np.NaN, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga":  2.530, "Ge": np.NaN, "As":  2.103, "Se": np.NaN, "Br":  np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "Se": {"H":   1.475, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":   1.676, "N":   1.652, "O":   1.639, "F": np.NaN, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si":  2.058, "P": np.NaN, "S":   2.037, "Cl": np.NaN, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge":  2.135, "As": np.NaN, "Se":  2.166, "Br":  np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},
    "Br": {"H":   1.414, 
           "Li":  2.170, "Be": np.NaN, 
           "B":   1.888, "C":   1.821, "N":   1.765, "O":   1.718, "F":  1.759, "Ne": np.NaN, 
           "Na":  2.502, "Mg": np.NaN, "Al":  2.295, "Si":  2.209, "P":  2.171, "S":  np.NaN, "Cl":  2.136, "Ar": np.NaN,
           "K":   2.821, "Ca":  2.594, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": 2.352, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  2.281, 
           "Sb": np.NaN, "Te": np.NaN, "I":  2.469},
    "Sb": {"H":  np.NaN, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":   1.835, "O":  np.NaN, "F":  1.918, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":  np.NaN, "Cl":  2.335, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  np.NaN, 
           "Sb": np.NaN, "Te": np.NaN, "I":  np.NaN},  
    "Te": {"H":   1.656, 
           "Li": np.NaN, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":   1.825, "F": np.NaN, "Ne": np.NaN, 
           "Na": np.NaN, "Mg": np.NaN, "Al": np.NaN, "Si": np.NaN, "P": np.NaN, "S":  np.NaN, "Cl": np.NaN, "Ar": np.NaN,
           "K":  np.NaN, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br": np.NaN, 
           "Sb": np.NaN, "Te":  2.557, "I":  np.NaN},    
    "I":  {"H":   1.609, 
           "Li":  2.392, "Be": np.NaN, 
           "B":  np.NaN, "C":  np.NaN, "N":  np.NaN, "O":  	1.868, "F": 1.910, "Ne": np.NaN, 
           "Na":  2.711, "Mg": np.NaN, "Al":  2.537, "Si": np.NaN, "P": np.NaN, "S":  np.NaN, "Cl":  2.321, "Ar": np.NaN,
           "K":   3.048, "Ca": np.NaN, "Sc": np.NaN, "Ti": np.NaN, "V": np.NaN, "Cr": np.NaN, "Cu": np.NaN, "Zn": np.NaN, "Ga": np.NaN, "Ge": np.NaN, "As": np.NaN, "Se": np.NaN, "Br":  2.469, 
           "Sb": np.NaN, "Te": np.NaN, "I":  2.665}
}                      

class NanoSim:
    """Class for Nanoreactor Simulation Results

    This class contains functions which enable a fully automated post-processing based on the generated trajectory
    and bond order files from ab initio nanoreactor simulations.

    Args:
        traj_file: trajectory file path\n
                   Format: 1 number of atoms\n
                           2 TIME: time step\n
                           3 elem x y z\n
                           4 elem x y z\n
                                 .\n
                                 .\n

        bond_order_file: Wiberg bond order file\n
                         Format: 1 TIME: time step\n
                                 2 wbo(0,1)\n
                                 3 wbo(0,2)\n
                                 4 wbo(0,3)\n
                                     .\n
                                     .\n
                         only upper triangular (without diagonal elements because equal to 0) is stored to reduce file size\n

        mols_file: atom indices fragment file (if it already exists to avoid unecessary computation)
        dt: MD time step
        read_rate: output rate used in the MD to allow reconstruction of the time scale during the post-processing
    """

    def __init__(
        self,
        traj_file: str,
        bond_order_file: str,
        pop_file: str,
        mols_file = None,
        dt: float = 0.5,
        read_rate: int = 50,
    ):
        self.dt = dt
        self.read_rate = read_rate

        self.atom_map, self.traj = read_traj_file(traj_file)
        self.pop = read_pop_file(pop_file)
        self.bond_orders = read_bo_file(bond_order_file, natoms=len(self.atom_map))
        if mols_file != None:
            self.fragments = read_frag_file(mols_file)
        else:
            self.fragments = []

        # Initialize other variables
        self.elem_fragments = []
        self.smiles = []
        self.mol_formulas = []
        self.df = pd.DataFrame()

    def calc_frags(
        self, ts: int, bond_order_matrix: np.ndarray, buffer_type: str = "Ar"
    ):
        """Group atoms into fragments based on atom indices for given time step. Bonds are considered above 0.5 for the (Wiberg) bond orders.
            If the WBO's of one atom do not exceed 0.5 then a second criterion based on relative bond lengths is employed.
            Buffer type needs to be specified, the default is argon.

        Args:
            ts: time step index
            bond_order_matrix: Numpy array containing complete bond order matrix for ts
            buffer_type: element used as buffer; default is argon
        Returns:
            fragments: list of lists containing the fragments based on atom indices for ts
        """

        global __std_bond_lengths__

        if len(bond_order_matrix) == 0:
            print("WBO required, when calculating fragments!")
            sys.exit(1)

        fragments = []
        natoms = len(self.atom_map)

        for i in range(0, natoms):
            for j in range(i + 1, natoms):
                if bond_order_matrix[i][j] < 0.5e0:
                    continue  # skip if no bond
                at1 = i + 1
                at2 = j + 1
                found = []
                count = 0

                for fragment in fragments:
                    if is_in(at1, fragment) and is_in(at2, fragment):
                        found.append(count)
                    elif is_in(at1, fragment):
                        fragment.append(at2)
                        found.append(count)
                    elif is_in(at2, fragment):
                        fragment.append(at1)
                        found.append(count)
                    count += 1

                if len(found) == 0:
                    fragments.append([at1, at2])
                elif len(found) > 1:
                    new_fragments = []
                    nfrag = len(found)
                    f = []
                    for nfrag_i in range(nfrag):
                        f.append(fragments[found[nfrag_i]])

                    fscr = f[0]
                    for nfrag_i in range(1, nfrag):
                        fscr = fscr + list(set(f[nfrag_i]) - set(fscr))

                    new_fragments.append(fscr)
                    for nfrag_i in range(len(fragments)):
                        if is_in(nfrag_i, found) == 0:
                            new_fragments.append(fragments[nfrag_i])

                    fragments = []
                    fragments = new_fragments

        for atom1 in range(1, natoms + 1):
            atom1_type = self.atom_map[atom1 - 1].capitalize()

            distance_array = np.empty(natoms)
            distance_array[:] = np.NaN

            sd_array = np.empty(natoms)
            sd_array[:] = np.NaN

            bool_check_atom1 = any(atom1 in sublist for sublist in fragments)
            if bool_check_atom1:
                continue
            elif atom1_type == buffer_type:
                fragments.append([atom1])
                continue

            for atom2 in range(1, natoms + 1):
                atom2_type = self.atom_map[atom2 - 1].capitalize()
                bool_check_atom2 = any(atom2 in sublist for sublist in fragments)
                if atom1 == atom2:
                    continue
                elif atom2_type == buffer_type:
                    if bool_check_atom2:
                        continue
                    else:
                        fragments.append([atom2])
                        continue
                
                if np.isnan(__std_bond_lengths__[atom1_type][atom2_type]):
                    pass
                else:
                    coord_atom1 = np.array(self.traj[ts][atom1 - 1])
                    coord_atom2 = np.array(self.traj[ts][atom2 - 1])
                    distance = np.linalg.norm(coord_atom1 - coord_atom2)

                    distance_array[atom2 - 1] = distance
                    distance_array[atom1 - 1] = np.NaN

                    sd_array[atom2 - 1] = np.sqrt(
                        pow(
                            distance_array[atom2 - 1]
                            - __std_bond_lengths__[atom1_type][atom2_type],
                            2,
                        )
                    )

            atom_partner = np.nanargmin(sd_array) + 1

            bool_check_atom_partner = any(
                atom_partner in sublist for sublist in fragments
            )

            if bool_check_atom_partner:
                for frag in fragments:
                    if atom_partner in frag:
                        frag.append(atom1)
            else:
                fragments.append([atom1, atom_partner])

        return fragments

    def generate_frag_lists(self):
        """Compute fragments for all time steps and write file containing fragment information.
            This can also be done on-the-fly during the MD simulation.

        Args:
            -
        Returns:
            -
        """

        natoms = len(self.atom_map)
        fragments = []

        for ts in range(len(self.bond_orders)):
            bond_order_matrix = self.bond_orders[ts]
            fragments = self.calc_frags(ts, bond_order_matrix)
            self.fragments.append(fragments)
            timestep = ts * self.dt * self.read_rate

            length = 0
            for frag_list in fragments:
                length += len(frag_list)

            if length != natoms:
                print(str(ts) + ": " + str(length))
            write_frag_file("mols.dat", timestep, fragments)

        return

    def mol_from_graphs(
        self, node_list: list, adjacency_matrix: np.ndarray, pop: np.ndarray
    ) -> Chem.rdchem.Mol:
        """Compute RDKit mol object from atom number list and adjacency matrix (WBO matrix in this case).

        Args:
            node_list: list of atomic numbers
            adjacency_matrix: WBO matrix for fragment of interest
            pop: numpy 2D array, first array are the Mulliken charges, second array are the Mulliken spin populations (alpha-beta)
        Returns:
            mol: RDKit mol object
        """

        # create empty editable mol object
        mol = Chem.rdchem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            a = Chem.rdchem.Atom(node_list[i])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        # add bonds between adjacent atoms
        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):

                # only traverse half the matrix
                if iy <= ix:
                    continue

                # add relevant bond type (there are many more of these)
                if bond == 0.0:
                    continue
                elif bond == 1.0:
                    bond_type = Chem.rdchem.BondType.SINGLE
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
                elif bond == 2.0:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
                elif bond == 3.0:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        # take care of charge and spin properties
        for atom in mol.GetAtoms():
            atom.SetNoImplicit(True)

        if np.abs(round(np.sum(pop[1]))) != 0:  # this is the case when there are radicals involved
            print("Found radicals")
            print(node_list)
            print(pop[1])
            no_radicals = int(np.abs(round(np.sum(pop[1]))))

            max_spin = int(np.argmax(np.abs(pop[1])))
            mol.GetAtomWithIdx(max_spin).SetNumRadicalElectrons(no_radicals)

        if round(np.sum(pop[0])) != 0:
            mol.UpdatePropertyCache(strict=False)
            total_charge = int(round(np.sum(pop[0])))
            if np.abs(total_charge) >= 2 and len(pop[0]) >= 2:
                total_charge /= abs(total_charge)
                total_charge = int(total_charge)
            for i in range(int(round(np.sum(pop[0])))):
                #metal_character = [int(check_electroneg(atom)) for atom in mol.GetAtoms()]
                metal_character = [int(is_metal(atom.GetAtomicNum())) for atom in mol.GetAtoms()]
                correct_val = [correct_valences(atom) for atom in mol.GetAtoms()]
                if np.sign(total_charge) in metal_character:
                    matching_atom_i = metal_character.index(np.sign(total_charge))
                    if not correct_val[matching_atom_i]:
                        matching_atom = mol.GetAtoms()[matching_atom_i]
                        matching_atom.SetFormalCharge(total_charge)
                        break
                elif False in correct_val:
                    matching_atom = mol.GetAtoms()[correct_val.index(False)]
                    matching_atom.SetFormalCharge(total_charge)
                    break
                else:
                    print("Charge was not assigned!")
                    
        # here I will try sth new
        mol = mol.GetMol()
        mol.UpdatePropertyCache(strict=False)
        #Chem.SanitizeMol()
        for at in mol.GetAtoms():
            at_number = at.GetAtomicNum()
            pt = Chem.GetPeriodicTable()
            possible_valences = pt.GetValenceList(at_number)
            charge = at.GetFormalCharge()
            total_valence = np.abs(at.GetExplicitValence() + at.GetImplicitValence() - charge + at.GetNumRadicalElectrons())
            if int(total_valence) not in possible_valences:
                if len(possible_valences) == 1:
                    electroneg = check_electroneg(at)
                    at.SetFormalCharge(charge + int(electroneg * (possible_valences[0] - total_valence)))
                else:
                    for i_val, val in enumerate(possible_valences):
                        val_dif = val - total_valence
                        if abs(val_dif) < 2:
                            electroneg = check_electroneg(at)
                            at.SetFormalCharge(charge + int(electroneg * val_dif))
                            break
                        elif -val_dif >= 2 and i_val == (len(possible_valences) - 1):
                            at.SetFormalCharge(charge + int(electroneg * val_dif))
                            break

        mol.UpdatePropertyCache(strict=False)
        mol = Chem.rdchem.RWMol(mol)
        mol = merge_neighbor_radicals(mol)
        mol = merge_neighbor_anions(mol)
        mol = mol.GetMol()        
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True)
        if mol.GetNumAtoms() != len(pop[0]):
            print("Too many atoms!!" + str(mol.GetNumAtoms()))

        #final_charge = 0
        #final_spin = 0
        #for atom in mol.GetAtoms():
        #    final_charge += atom.GetFormalCharge()
        #    final_spin += atom.GetNumRadicalElectrons()
        
        #if final_charge != round(np.sum(pop[0])):
        #    print("Charge differs from the computed!")
        #    print(final_charge)
        #    print(round(np.sum(pop[0])))
        #elif final_spin != round(np.sum(pop[1])):
        #    print("Spin state differs from the computed!")
        #    print(final_spin)
        #    print(round(np.sum(pop[1])))           
            
        return mol

    def generate_mols_smiles(self):
        """Compute SMILES and corresponding molecular formulas from lists of fragments and bond order matrix.

        Args:
            pop_path: Path to the numpy archive
        Returns:
            -
        """

        #pop_archive = np.load(pop_path, allow_pickle=True)
        #pop = [pop_archive[key] for key in pop_archive]

        for ts in range(len(self.fragments)):
            self.elem_fragments.append([])
            for i_frag, frag in enumerate(self.fragments[ts]):
                self.elem_fragments[ts].append([])
                for i in frag:
                    self.elem_fragments[ts][i_frag].append(self.atom_map[int(i) - 1])

        natoms = len(self.atom_map)
        mols = []
        
        for ts in range(len(self.fragments)):
            bond_order_matrix = round_bond_order_numpy(self.bond_orders[ts])
            #bond_order_matrix = np.zeros((natoms, natoms))
            self.smiles.append([])
            self.mol_formulas.append([])
            mols.append([])
            '''
            for i in range(natoms):
                for j in range(i + 1, natoms):

                    n = natoms

                    bond_order_matrix[i][i] = 0

                    index = int(
                        (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1
                    )

                    bond_order_matrix[i][j] = round_bond_order_numpy(bond_order_ts[index])
                    bond_order_matrix[j][i] = bond_order_matrix[i][j]
            '''
            # Go through bond orders list and write for every fragment the correct bond order matrix and then convert it to SMILES

            for fragment_index in range(len(self.fragments[ts])):
                bond_order_matrix_fragment = np.zeros(
                    (
                        len(self.fragments[ts][fragment_index]),
                        len(self.fragments[ts][fragment_index]),
                    )
                )

                pop_fragment = np.zeros((2, len(self.fragments[ts][fragment_index])))

                for i in range(len(self.fragments[ts][fragment_index])):
                    atom_index1 = int(self.fragments[ts][fragment_index][i]) - 1
                    pop_fragment[0][i] = self.pop[ts][1][atom_index1]
                    if np.shape(self.pop)[1] == 3: # npy file contains both charge and spin populations
                        pop_fragment[1][i] = self.pop[ts][2][atom_index1]
                    else:
                        pop_fragment[1][i] = 0.e0 # when mult for the restricted SCF was 1

                    for j in range(i + 1, len(self.fragments[ts][fragment_index])):
                        atom_index2 = int(self.fragments[ts][fragment_index][j]) - 1

                        bond_order_matrix_fragment[i][i] = 0
                        bond_order_matrix_fragment[i][j] = bond_order_matrix[
                            atom_index1
                        ][atom_index2]
                        bond_order_matrix_fragment[j][i] = bond_order_matrix_fragment[
                            i
                        ][j]

                atoms = [
                    int_atom(atom) for atom in self.elem_fragments[ts][fragment_index]
                ]

                # get mol and smiles
                mol = self.mol_from_graphs(
                    atoms, bond_order_matrix_fragment, pop_fragment
                )
                smile = rdmolfiles.MolToSmiles(mol)

                # transform smiles to mol again and back for uniformization and avoid adjustHs
                params = rdmolfiles.SmilesParserParams()
                params.removeHs=False
                params.sanitize = False
                mol = rdmolfiles.MolFromSmiles(smile, params)
                mol.UpdatePropertyCache(strict=False)
                rdmolops.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True)

                smile = rdmolfiles.MolToSmiles(mol)

                # calculate formula and add it to df
                formula = CalcMolFormula(mol)
                self.mol_formulas[ts].append(formula)

                # add new mol to mols list
                mols[ts].append(mol)
                self.smiles[ts].append(smile.split("."))

        return

    def generate_df(self):
        """Generate data frame after the evaluation of a nanoreactor simulation is finished.

        Args:
            -
        Returns:
            -
        """

        column_names = [
            "Time step [fs]",
            "# Fragment",
            "# Atom in Fragment",
            "Mols in Fragment",
            "XYZ",
            "SMILES",
            "Molecular Formulas",
        ]
        self.df = pd.DataFrame(columns=column_names)

        # Make some transformations

        # Time step list in fs
        timesteps = []
        for ts_i in range(len(self.traj)):
            timesteps.append(ts_i * self.dt * self.read_rate)

        # Divide trajectory according to fragments
        xyz_divided = divide_traj(self.fragments, self.traj)

        # Construct lists with fragment indices
        frag_indices = []
        for ts in range(len(self.fragments)):
            frag_indices.append([])
            for i_frag, frag in enumerate(self.fragments[ts]):
                frag_indices[ts].append(i_frag + 1)

        str_fragments = []
        for ts in range(len(self.fragments)):
            str_fragments.append([])
            for frag_index, frag in enumerate(self.fragments[ts]):
                str_fragments[ts].append([])
                for atom_index in frag:
                    str_fragments[ts][frag_index].append(str(atom_index))

        # Fill in dataframe
        self.df["Time step [fs]"] = timesteps
        self.df["# Atom in Fragment"] = str_fragments
        self.df["# Fragment"] = frag_indices
        self.df["Mols in Fragment"] = self.elem_fragments
        self.df["XYZ"] = xyz_divided
        self.df["SMILES"] = self.smiles
        self.df["Molecular Formulas"] = self.mol_formulas

        index = 0
        for i in range(len(self.smiles)):
            for j in range(len(self.smiles[i])):
                if self.smiles[i][j] == ["Revise structure"]:
                    print("Time step: %.1i" % i)
                    print("Fragment index: %1i" % j)
                    index += 1
        print("\n--------Finished molecular transformations--------")
        print("There are %.1i molecules in total to be revised.\n" % index)

        self.df.to_csv("dataframe_nanosim.csv", sep=";")

        return
