import sys
import numpy as np

# list of element symbols
global __elements__
__elements__ = ['h',  'he',
                'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne',
                'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar',
                'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
                'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe',
                'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w', 're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
                'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu', 'am', 'cm', 'bk', 'cf', 'es', 'fm', 'md', 'no', 'lr', 'rf', 'db', 'sg', 'bh', 'hs', 'mt', 'ds', 'rg', 'cn', 'nh', 'fl', 'mc', 'lv', 'ts', 'og']

def is_in(elem, my_list: list) -> bool:
    ''' Check if element is in list.

    Args:
        elem: element to be checked for
        my_list: list to look in
    Returns:
        found: boolean stating if element has been found or not
    '''
    l = [x for x in my_list if x==elem]
    found = len(l) > 0 
    
    return found

def str_atom(atom: int) -> str:
    ''' Find element symbol based on atomic number.

    Args:
        atom: atomic number
    Returns:
        atom: element symbol
    '''
    global __elements__
    atom = __elements__[atom-1]
    
    return atom

def int_atom(atom: str) -> int:
    ''' Find atomic number based on element symbol.

    Args:
        atom: element symbol
    Returns:
        atom: atomic number
    '''
    
    global __elements__
    atom = atom.lower()
    
    return __elements__.index(atom) + 1

def round_bond_order(bo: float) -> float:
    ''' Round bond orders to integers.

    Args:
        bo: calculated (Wiberg) bond order
    Returns:
        newbo: rounded (Wiberg) bond order
    '''
    if bo >= 3.5:
        newbo = 0.0
    elif bo >= 2.5:
        newbo = 3.0
    elif bo >= 1.5:
        newbo = 2.0
    elif bo >= 0.5:
        newbo = 1.0
    else:
        newbo = 0.0

    return newbo

def divide_traj(atom_indices_frag: list, xyz_list: list) -> list:
    ''' Divide trajectory into small trajectory of every fragment.

    Args:
        bo: calculated (Wiberg) bond order
    Returns:
        newbo: rounded (Wiberg) bond order
    '''
    xyz_divided = []
    for ts in range(len(atom_indices_frag)):
        xyz_divided.append([])
        for mol_index in range(len(atom_indices_frag[ts])):
            xyz_divided[ts].append([])
            for atom_index in range(len(atom_indices_frag[ts][mol_index])):
                xyz_divided[ts][mol_index].append(xyz_list[ts][int(atom_indices_frag[ts][mol_index][atom_index])-1])
    
    return xyz_divided

def sort_merge_str_list_to_int_list(str_list_list: list) -> list:
    ''' Transform a list of lists of strings into a sorted integer list.

    Args:
        str_list_list: list of lists containing strings which represent numbers
    Returns:
        int_list_merged: sorted and merged list of integers
    '''
    str_list_list_merged = sum(str_list_list,[])
    int_list_merged = list(map(int, str_list_list_merged))
    int_list_merged.sort()
    return int_list_merged