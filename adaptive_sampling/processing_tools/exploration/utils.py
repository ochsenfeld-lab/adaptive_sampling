from rdkit import Chem
import rdkit.Chem.rdmolops as rdmolops
import json
import pandas as pd
import numpy as np
from typing import Tuple
import os

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
    """Check if element is in list.

    Args:
        elem: element to be checked for
        my_list: list to look in
    Returns:
        found: boolean stating if element has been found or not
    """
    l = [x for x in my_list if x == elem]
    found = len(l) > 0

    return found


def str_atom(atom: int) -> str:
    """Find element symbol based on atomic number.

    Args:
        atom: atomic number
    Returns:
        atom: element symbol
    """
    global __elements__
    atom_str = __elements__[atom - 1]

    return atom_str


def int_atom(atom: str) -> int:
    """Find atomic number based on element symbol.

    Args:
        atom: element symbol
    Returns:
        atom: atomic number
    """

    global __elements__
    atom = atom.lower()

    return __elements__.index(atom) + 1

def round_bond_order_numpy(bond_orders: np.ndarray) -> np.ndarray:
    """Round bond orders to integers using vectorized operations.

    Args:
        bond_orders: 2D NumPy array of calculated (Wiberg) bond orders
    Returns:
        rounded_bond_orders: 2D NumPy array of rounded (Wiberg) bond orders
    """
    rounded_bond_orders = np.zeros_like(
        bond_orders
    )  # Erstellt ein Array mit derselben Form
    rounded_bond_orders[bond_orders >= 3.5] = 0.0
    rounded_bond_orders[(bond_orders >= 2.5) & (bond_orders < 3.5)] = 3.0
    rounded_bond_orders[(bond_orders >= 1.5) & (bond_orders < 2.5)] = 2.0
    rounded_bond_orders[(bond_orders >= 0.5) & (bond_orders < 1.5)] = 1.0
    rounded_bond_orders[bond_orders < 0.5] = 0.0

    return rounded_bond_orders

def round_bond_order(bo: float) -> float:
    """Round bond orders to integers.

    Args:
        bo: calculated (Wiberg) bond order
    Returns:
        newbo: rounded (Wiberg) bond order
    """

    if bo >= 3.5:
        newbo = 0.0
    elif bo >= 2.5:
        newbo = 3.0
    elif bo >= 1.5:
        newbo = 2.0
    #elif bo >= 1.25:
    #    newbo = 1.5
    elif bo >= 0.5:
        newbo = 1.0
    else:
        newbo = 0.0

    return newbo


def divide_traj(atom_indices_frag: list, xyz_list: list) -> list:
    """Divide trajectory into small trajectory of every fragment.

    Args:
        bo: calculated (Wiberg) bond order
    Returns:
        newbo: rounded (Wiberg) bond order
    """
    xyz_divided = []
    for ts in range(len(atom_indices_frag)):
        xyz_divided.append([])
        for mol_index in range(len(atom_indices_frag[ts])):
            xyz_divided[ts].append([])
            for atom_index in range(len(atom_indices_frag[ts][mol_index])):
                xyz_divided[ts][mol_index].append(
                    xyz_list[ts][int(atom_indices_frag[ts][mol_index][atom_index]) - 1]
                )

    return xyz_divided


def sort_merge_str_list_to_int_list(str_list_list: list) -> list:
    """Transform a list of lists of strings into a sorted integer list.

    Args:
        str_list_list: list of lists containing strings which represent numbers
    Returns:
        int_list_merged: sorted and merged list of integers
    """
    str_list_list_merged = sum(str_list_list, [])
    int_list_merged = list(map(int, str_list_list_merged))
    int_list_merged.sort()
    return int_list_merged


def merge_neighbor_radicals(mol: Chem.rdchem.RWMol) -> Chem.rdchem.RWMol:
    """Find radical sites where the BO analysis has actually yielded a single bond instead of a double bond.
    Args:
        mol: molecule to search in
    Returns:
        (rad1, rad2): tuple of neigboring radical sites
    """

    for at_ID1 in range(len(mol.GetAtoms())):
        atom1 = mol.GetAtoms()[at_ID1]

        # return radicals_dict
        for at_ID2 in range(at_ID1 + 1, len(mol.GetAtoms())):
            #atom1 = mol.GetAtoms()[at_ID1]
            atom2 = mol.GetAtoms()[at_ID2]
            if mol.GetBondBetweenAtoms(at_ID1, at_ID2) != None:
                if (
                    atom1.GetNumRadicalElectrons() == atom2.GetNumRadicalElectrons()
                    and atom1.GetNumRadicalElectrons() > 0
                ):
                    no_radicals = atom1.GetNumRadicalElectrons()
                    if no_radicals == 1:
                        if (
                            mol.GetBondBetweenAtoms(at_ID1, at_ID2).GetBondType()
                            == Chem.rdchem.BondType.SINGLE
                        ):
                            mol.RemoveBond(at_ID1, at_ID2)
                            mol.AddBond(at_ID1, at_ID2, Chem.rdchem.BondType.DOUBLE)
                            atom1.SetNumRadicalElectrons(0)
                            atom2.SetNumRadicalElectrons(0)
                        elif (
                            mol.GetBondBetweenAtoms(at_ID1, at_ID2).GetBondType()
                            == Chem.rdchem.BondType.DOUBLE
                        ):
                            mol.RemoveBond(at_ID1, at_ID2)
                            mol.AddBond(at_ID1, at_ID2, Chem.rdchem.BondType.TRIPLE)
                            atom1.SetNumRadicalElectrons(0)
                            atom2.SetNumRadicalElectrons(0)
                    elif no_radicals == 2:
                        if (
                            mol.GetBondBetweenAtoms(at_ID1, at_ID2).GetBondType()
                            == Chem.rdchem.BondType.SINGLE
                        ):
                            mol.RemoveBond(at_ID1, at_ID2)
                            mol.AddBond(at_ID1, at_ID2, Chem.rdchem.BondType.TRIPLE)
                            atom1.SetNumRadicalElectrons(0)
                            atom2.SetNumRadicalElectrons(0)
            else:
                pass
    return mol

def merge_neighbor_anions(mol: Chem.rdchem.RWMol) -> Chem.rdchem.RWMol:
    """Find radical sites where the BO analysis has actually yielded a single bond instead of a double bond.
    Args:
        mol: molecule to search in
    Returns:
        (rad1, rad2): tuple of neigboring radical sites
    """

    for at_ID1 in range(len(mol.GetAtoms())):
        atom1 = mol.GetAtoms()[at_ID1]

        # return radicals_dict
        for at_ID2 in range(at_ID1 + 1, len(mol.GetAtoms())):
            atom1 = mol.GetAtoms()[at_ID1]
            atom2 = mol.GetAtoms()[at_ID2]
            if mol.GetBondBetweenAtoms(at_ID1, at_ID2) != None:
                if (
                    atom1.GetFormalCharge() == atom2.GetFormalCharge()
                    and atom1.GetFormalCharge() < 0
                ):
                    charge = atom1.GetFormalCharge()
                    #print(charge)
                    if charge == -1:
                        if (
                            mol.GetBondBetweenAtoms(at_ID1, at_ID2).GetBondType()
                            == Chem.rdchem.BondType.SINGLE
                        ):
                            mol.RemoveBond(at_ID1, at_ID2)
                            mol.AddBond(at_ID1, at_ID2, Chem.rdchem.BondType.DOUBLE)
                            atom1.SetFormalCharge(0)
                            atom2.SetFormalCharge(0)
                        elif (
                            mol.GetBondBetweenAtoms(at_ID1, at_ID2).GetBondType()
                            == Chem.rdchem.BondType.DOUBLE
                        ):
                            mol.RemoveBond(at_ID1, at_ID2)
                            mol.AddBond(at_ID1, at_ID2, Chem.rdchem.BondType.TRIPLE)
                            atom1.SetFormalCharge(0)
                            atom2.SetFormalCharge(0)

            else:
                pass
    return mol

def normal_sanitize(mol: Chem.rdchem.RWMol):
    mol.UpdatePropertyCache(strict=False)
    #rdmolops.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_KEKULIZE|
    #                     Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
    #                     Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
    #                     catchErrors=True)

    rdmolops.SanitizeMol(mol)

    return


def special_sanitize(mol: Chem.rdchem.RWMol):
    mol.UpdatePropertyCache(strict=False)
    for atom in mol.GetAtoms():
        no_bonds = atom.GetExplicitValence()
        pt = Chem.GetPeriodicTable()  # type: ignore
        minv = min(pt.GetValenceList(atom.GetAtomicNum()))
        maxv = max(pt.GetValenceList(atom.GetAtomicNum()))

        #print("atomic number:" + str(atom.GetAtomicNum()))
        #print(no_bonds)
        #print(minv)
        #print(maxv)
        
        if no_bonds < minv:
            atom.SetFormalCharge(-(minv - no_bonds))
        elif no_bonds > maxv:
            atom.SetFormalCharge(no_bonds - maxv)

    mol.UpdatePropertyCache(strict=False)
    for atom in mol.GetAtoms():
        if correct_valences(atom):
            rdmolops.SanitizeMol(mol)
        else:
            special_sanitize(mol)
    return

def correct_valences(atom):
    pt = Chem.GetPeriodicTable()
    
    absolute_valence = np.abs(atom.GetExplicitValence() + atom.GetImplicitValence() + atom.GetNumRadicalElectrons() - atom.GetFormalCharge())
    if int(absolute_valence) in pt.GetValenceList(atom.GetAtomicNum()): 
        return True
    else:
        return False

def is_metal(at_number: int) -> float:
    non_metals = [1, 2, 6, 7, 8, 9, 10, 15, 16, 17, 18, 34, 35, 36, 53, 54, 86]
    metalloids = [5, 14, 32, 33, 51, 52, 85]
    if at_number in non_metals:
        return -1.e0
    else:
        return +1.e0 
       
def check_electroneg(atom) -> float:
    pt = Chem.GetPeriodicTable()
    at_number = atom.GetAtomicNum()
    at_period = pt.GetNOuterElecs(at_number)
    
    electroneg = +1
    for neigh in atom.GetNeighbors():
        neigh_number = neigh.GetAtomicNum()
        neigh_period = pt.GetNOuterElecs(neigh_number)
        
        if at_period >= neigh_period and at_number < neigh_number:
            electroneg = -1
            break
        elif at_period > neigh_period:
            electroneg = -1 
            break

    return electroneg


# function to find fragment index for atom and ts
def find_reactions(
    df: pd.DataFrame, atom_r: str, ts: int, step: int = 80
) -> Tuple[list, list, list]:
    """Find reaction by comparing fragments containing given atom index at time step ts and ts + step.

    Args:
        df: data frame containing all necessary information from nanoreactor simulation
        atom_r: atom index of reactive atom to be searched for
        ts: reactant time step
        step: interval between reactants and products

    Returns:
        atom_indices_reactant_list_sorted: list of sorted atom indices present in the reaction
        [atom_indices_reactant_list, atom_indices_product_list]: list containing the reaction expressed with atom indices
        [smiles_reactant_list, smiles_product_list]: list containing the reaction expressed in SMILES
    """
    atom_indices_reactant_list = []
    atom_indices_product_list = []
    smiles_reactant_list = []
    smiles_product_list = []

    # find product
    for fragment in range(len(df["# Fragment"][ts + step])):
        if atom_r in df["# Atom in Fragment"][ts + step][fragment]:
            atom_indices_product_list.append(
                df["# Atom in Fragment"][ts + step][fragment]
            )
            for elem in df["SMILES"][ts + step][fragment]:
                smiles_product_list.append(elem)
            break

    # look up atoms in the product and add fragments
    for atom_index in atom_indices_product_list[0]:
        for fragment in range(len(df["# Fragment"][ts])):
            if atom_index in df["# Atom in Fragment"][ts][fragment]:
                if (
                    df["# Atom in Fragment"][ts][fragment]
                    not in atom_indices_reactant_list
                ):
                    atom_indices_reactant_list.append(
                        df["# Atom in Fragment"][ts][fragment]
                    )
                    for elem in df["SMILES"][ts][fragment]:
                        smiles_reactant_list.append(elem)

    # return early if no reaction found
    if smiles_reactant_list == smiles_product_list:
        return ([], [], [])

    # transform atom_indices_lists into merged lists of sorted integers
    atom_indices_reactant_list_sorted = sort_merge_str_list_to_int_list(
        atom_indices_reactant_list
    )
    atom_indices_product_list_sorted = sort_merge_str_list_to_int_list(
        atom_indices_product_list
    )

    # look for missed atoms in the reactant and product lists in a loop
    # maybe use set instead of list to avoid sort
    while atom_indices_reactant_list_sorted != atom_indices_product_list_sorted:
        for reactant_index in range(len(atom_indices_reactant_list)):
            for atom_index in atom_indices_reactant_list[reactant_index]:
                for fragment in range(len(df["# Fragment"][ts + step])):
                    if atom_index in df["# Atom in Fragment"][ts + step][fragment]:
                        if (
                            df["# Atom in Fragment"][ts + step][fragment]
                            not in atom_indices_product_list
                        ):
                            atom_indices_product_list.append(
                                df["# Atom in Fragment"][ts + step][fragment]
                            )
                            for elem in df["SMILES"][ts + step][fragment]:
                                smiles_product_list.append(elem)

                            atom_indices_product_list_sorted = (
                                sort_merge_str_list_to_int_list(
                                    atom_indices_product_list
                                )
                            )
                            if (
                                atom_indices_reactant_list_sorted
                                == atom_indices_product_list_sorted
                            ):
                                break

            for product_index in range(len(atom_indices_product_list)):
                for atom_index in atom_indices_product_list[product_index]:
                    for fragment in range(len(df["# Fragment"][ts])):
                        if atom_index in df["# Atom in Fragment"][ts][fragment]:
                            if (
                                df["# Atom in Fragment"][ts][fragment]
                                not in atom_indices_reactant_list
                            ):
                                atom_indices_reactant_list.append(
                                    df["# Atom in Fragment"][ts][fragment]
                                )
                                for elem in df["SMILES"][ts][fragment]:
                                    smiles_reactant_list.append(elem)

                                atom_indices_reactant_list_sorted = (
                                    sort_merge_str_list_to_int_list(
                                        atom_indices_reactant_list
                                    )
                                )
                                if (
                                    atom_indices_reactant_list_sorted
                                    == atom_indices_product_list_sorted
                                ):
                                    break

    # exclude atom switches
    if smiles_reactant_list == smiles_product_list:
        return ([], [], [])

    return (
        atom_indices_reactant_list_sorted,
        [atom_indices_reactant_list, atom_indices_product_list],
        [smiles_reactant_list, smiles_product_list],
    )


def construct_reactions_list(
    df: pd.DataFrame, start_ts_index: int = 19, period_ts_steps: int = 80
) -> list:
    """Get list of reactions to be able to construct network.

    Args:
        df: data frame containing all necessary information from nanoreactor simulation
        start_ts_index: specify index of time step at which the first search should be conducted
                        Per default this is set for the smooth step spherical constraint function at the end of the expansion period.
        period_ts_steps: step width to search for products, corresponds to period of the confinement function
    Returns:
        reactions_list: list of reactions\n
                Format: [event #, [ts_r, ts_p], [smiles_r...], [smiles_p...]]\n
    """

    # ts # = 19 --> always look at steps at the end of the expansion (sin_cos: 19*50*0.5=475, at 500 fs the contraction starts)
    events = []
    time_steps = []
    reactions = []
    reactions_list = []
    atom_indices = []

    natoms = sum([len(listElem) for listElem in df["# Atom in Fragment"][0]])
    # print(natoms)

    event_counter = 0

    # first event
    atom_list_sorted = []
    print("Time Step: " + str(0) + " -> " + str(start_ts_index))
    for atom_index in range(natoms):
        if atom_index + 1 not in atom_list_sorted:
            try:
                atom_list_sorted, atom_index_list, smiles_reaction = find_reactions(
                    df, str(atom_index + 1), 0, start_ts_index
                )
               
                set_reactants = set(smiles_reaction[0])
                set_products = set(smiles_reaction[1])
                if [[0, start_ts_index], set_reactants, set_products, atom_list_sorted] not in reactions:
                #if [set_reactants, set_products] not in reactions:
                    event_counter += 1
                    events.append(event_counter)
                    time_steps.append([0, start_ts_index])
                    reactions.append([set_reactants, set_products])
                    reactions.append([[0, start_ts_index], set_reactants, set_products, atom_list_sorted])
                    atom_indices.append(atom_index_list)

                    reactions_list.append(
                        [
                            "Event " + str(event_counter),
                            [0, start_ts_index],
                            smiles_reaction[0],
                            smiles_reaction[1],
                            atom_list_sorted
                        ]
                    )
                    print("# Event: " + str(event_counter))
                    print(str(smiles_reaction[0]) + " -> " + str(smiles_reaction[1]))

            except IndexError:
                pass

    for ts in range(
        start_ts_index, len(df["Time step [fs]"]) - period_ts_steps, period_ts_steps
    ):
        atom_list_sorted = []
        print("Time Step: " + str(ts) + " -> " + str(ts + period_ts_steps))
        for atom_index in range(natoms):
            if atom_index + 1 not in atom_list_sorted:
                try:
                    atom_list_sorted, atom_index_list, smiles_reaction = find_reactions(
                        df, str(atom_index + 1), ts, period_ts_steps
                    )
                    set_reactants = set(smiles_reaction[0])
                    set_products = set(smiles_reaction[1])
                    #if [set_reactants, set_products] not in reactions:
                    if [[ts, ts + period_ts_steps], set_reactants, set_products, atom_list_sorted] not in reactions:
                        event_counter += 1
                        events.append(event_counter)
                        time_steps.append([ts, ts + period_ts_steps])
                        #reactions.append([set_reactants, set_products])
                        reactions.append([[ts, ts + period_ts_steps], set_reactants, set_products, atom_list_sorted])
                        atom_indices.append(atom_index_list)

                        reactions_list.append(
                            [
                                "Event " + str(event_counter),
                                [ts, ts + period_ts_steps],
                                smiles_reaction[0],
                                smiles_reaction[1],
                                atom_list_sorted
                            ]
                        )
                        print("# Event: " + str(event_counter))
                        print(
                            str(smiles_reaction[0]) + " -> " + str(smiles_reaction[1])
                        )

                except IndexError:
                    pass
    print("Number of events found: %3i" % (event_counter))

    with open("reactions_list.json", "w") as fp:
        json.dump(reactions_list, fp)

    return reactions_list

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in sorted(filenames):
            yield os.path.abspath(os.path.join(dirpath, f))

