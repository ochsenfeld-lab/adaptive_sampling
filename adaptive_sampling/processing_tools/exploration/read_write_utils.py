import pandas as pd
import ast
import json
from typing import Tuple
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.rdmolops as rdmolops
import rdkit.Chem.rdmolfiles as rdmolfiles
from ase.io import read, write, Trajectory

# read and write  all kinds of file types

def read_traj_file(ase_file: str) -> Tuple[list, list]:
    """Read ASE trajectory file.
    
    Args:
        ase_file: path to trajectory file in ASE .traj format

    Returns:
        atom_map: mapping of indices to atomic symbols
        xyz_list: complete trajectory (xyz coordinates)
    """

    atom_map = []
    xyz_list = []
    traj = read(ase_file, index=":")
    natoms = len(traj[0]) # type: ignore
    for ts, frame in enumerate(traj):
        xyz_list.append([])
        for i_at in range(natoms):
            if ts == 0:
                atom_map.append(frame[i_at].symbol) # type: ignore
            xyz_list[ts].append(frame[i_at].position) # type: ignore

    return atom_map, xyz_list

def read_xyz_traj(xyz_file: str) -> Tuple[list, list]:
    """Read trajectory file.

    Args:
        xyz_file: path to trajectory file from nanoreactor simulation\n
                  Format: 1 number of atoms\n
                          2 TIME: time step\n
                          3 elem x y z\n
                          4 elem x y z\n
                                .\n
                                .\n
    Returns:
        atom_map: order of atomic symbols at each time step
        xyz_list: complete trajectory as list of lists
    """

    index = -1
    atom_map_total = []
    xyz_list = []

    traj_file = open(xyz_file, "r")

    for line in traj_file:
        try:
            vals = line.strip().split()
        except:
            raise Exception("Failed to parse line...")

        if len(vals) == 1:
            pass
        elif len(vals) == 2:
            index += 1
            atom_map_total.append([])
            xyz_list.append([])
        elif len(vals) == 4:
            atom_map_total[index].append(vals[0])
            xyz_list[index].append([float(vals[1]), float(vals[2]), float(vals[3])])

    atom_map = atom_map_total[0]
    traj_file.close()

    return atom_map, xyz_list

def read_pop_file(pop_file: str) -> list:
    pop_archive = np.load(pop_file, allow_pickle=True)
    pop = [pop_archive[key] for key in pop_archive]

    return pop

def read_bo_file(bo_file: str, natoms: int) -> list:
    bo_archive = np.load(bo_file, allow_pickle=True)
    bond_order_indexed = [bo_archive[key] for key in bo_archive]
    timesteps = len(bond_order_indexed)
    bo_matrices_list = []
    for t in range(timesteps):
        bond_order_matrix = np.zeros((natoms, natoms))
        for bo in bond_order_indexed[t][1]:
            bond_order_matrix[int(bo[1]), int(bo[2])] = bo[0]
            bond_order_matrix[int(bo[2]), int(bo[1])] = bo[0]
        bo_matrices_list.append(bond_order_matrix)
    return bo_matrices_list

def read_frag_file(dat_file: str) -> list:
    """Read fragment file containing indices belonging to found molecules in each step.

    Args:
        dat_file: filepath to file containing (on-the-fly) computed fragments stored as lists of atom indices (starting at 1)\n
                  Format: 1  time step\n
                          2 at_index1   at_index2\n
                          3 at_index3   at_index4   at_index5\n
                          4 at_index6\n
                                .\n
                                .\n
    Returns:
        atom_indices_frag: list of lists storing corresponding list of atom indices for each fragment at every time step
    """

    atom_indices_frag = []
    index = -1

    mols_file = open(dat_file, "r")

    for line in mols_file:
        if line.startswith(" "):
            index += 1
            atom_indices_frag.append([])

        else:
            numbers = line.split()
            atom_indices_frag[index].append(numbers)

    mols_file.close()
    return atom_indices_frag


def write_frag_file(name: str, timestep: float, fragments: list):
    """Write fragment file containing indices belonging to found molecules in each step.

    Args:
        timestep: time step in fs
        name: name of file to be written
        fragments: found molecules as lists of atom indices\n
                  Format: 1  time step\n
                          2 at_index1   at_index2\n
                          3 at_index3   at_index4   at_index5\n
                          4 at_index6\n
                                .\n
                                .\n
    Returns:
        -
    """

    if timestep == 0.0:
        f = open(name, "w")
    else:
        f = open(name, "a")
    string = str("%20.10e\n") % (timestep)
    f.write(string)
    for i in range(len(fragments)):
        for j in fragments[i]:
            string = str("%i\t") % j
            f.write(string)
        string = str("\n")
        f.write(string)
    f.close()

    return


def read_trafo_df(df_file: str) -> pd.DataFrame:
    """Restore data frame containing all necessary information from the post-processing from file.

    Args:
        df_file: filepath to .csv file where data frame was stored (sep: ';')

    Returns:
        df : pd.DataFrame object with columns 'Time Step', '# Fragment', '# Atom in Fragment', '# Elem in Fragment', 'XYZ', 'SMILES', 'Molecular Formulas'
    """

    df = pd.read_csv(df_file, sep=";")

    new_fragments = []
    for ts in range(len(df["# Fragment"])):
        new_fragments.append(ast.literal_eval(df["# Fragment"][ts]))

    new_atom_in_fragments = []
    for ts in range(len(df["# Atom in Fragment"])):
        new_atom_in_fragments.append(ast.literal_eval(df["# Atom in Fragment"][ts]))

    new_SMILES = []
    for ts in range(len(df["SMILES"])):
        new_SMILES.append(ast.literal_eval(df["SMILES"][ts]))

    new_molecular_formulas = []
    string = []
    for ts in range(len(df["Molecular Formulas"])):
        string = df["Molecular Formulas"][ts].replace("'", "")
        string = string.replace("[", "")
        string = string.replace("]", "")
        string = string.replace(" ", "")
        new_molecular_formulas.append(string.split(","))

    df["# Fragment"] = new_fragments
    df["# Atom in Fragment"] = new_atom_in_fragments
    df["SMILES"] = new_SMILES
    df["Molecular Formulas"] = new_molecular_formulas

    return df


def read_reaction_list(json_file: str) -> list:
    """Read JSON file containing reaction data to construct network.

    Args:
        json_file: JSON file path where the reaction list was been stored

    Returns:
        reactions_list: list of reactions\n
                        Format: [event #, [ts_r, ts_p], [smiles_r...], [smiles_p...]]\n
    """
    with open(json_file, "r") as fp:
        reactions_list = json.load(fp)

    return reactions_list

def get_reaction_traj(traj_file: str, sim_df_file: str, reactions_list_file: str, event_no: list, path: str = "rct_traj", all_steps: bool = False):
    
    sim_df = read_trafo_df(sim_df_file)
    atom_map, xyz = read_xyz_traj(traj_file)
    reactions_list = read_reaction_list(reactions_list_file)
    
    # first, get desired reaction(s) and time steps
    timesteps = []
    reactants = []
    products = []
    atom_indices_sorted = []
    
    event_no.sort()
    for event in event_no:
        for ts in reactions_list[event - 1][1]:
            if ts not in timesteps:
                timesteps.append(ts)
                
        for react in reactions_list[event - 1][2]:
            if react not in reactants:
                reactants.append(react)
                
        for prod in reactions_list[event - 1][3]:
            if prod not in products:
                products.append(prod)
        
#        print(reactions_list[event - 1][4])
        for atom_index in reactions_list[event - 1][4]:
            if atom_index not in atom_indices_sorted:
                atom_indices_sorted.append(atom_index)

    #print(atom_indices_sorted)
        #atom_indices.extend(reactions_list[event-1][4])
    
    t_start = min(timesteps)
    t_end = max(timesteps)

    # get XYZ between the time steps and write file
    f_traj = open(path + "_" + str(event_no[0]) + "_" + str(event_no[-1]) + ".xyz", "a+")
    f_spin_mult = open(path + "_" + str(event_no[0]) + "_" + str(event_no[-1]) + ".txt", "a+")

    #total_charge = np.array(len(ts_range))
    #total_spin = 0

    atom_indices_frags = []

    if not all_steps:
        ts_range = [t_start, t_end]
    else:
        ts_range = np.arange(min(timesteps),max(timesteps))

    total_charge = np.zeros_like(ts_range)
    total_spin = np.zeros_like(ts_range)

    #ts_range.reverse()
    frags_i = np.empty((len(ts_range),0), dtype=object)
    #frags_i.tolist()
    #frags_i = frags_i.fill([])
    frags_i = frags_i.tolist()
    #print(type(frags_i))
    #frags_i[...]=[[] for _ in range(len(ts_range))]

    for frag_i, frag in enumerate(sim_df['# Atom in Fragment'][ts_range[0]]):
        for atom_index in atom_indices_sorted:
            int_frag = [ast.literal_eval(i) for i in frag]
            if atom_index in int_frag and frag_i not in frags_i[0]:
                frags_i[0].append(frag_i)
                
        #print(frags_i)

        for i_frag in frags_i[0]:
            int_frag = [ast.literal_eval(i) for i in sim_df['# Atom in Fragment'][ts_range[0]][i_frag]]
            for i_atom in int_frag:
                if i_atom not in atom_indices_frags:
                    atom_indices_frags.append(i_atom)

    no_atoms_old = 0
    no_atoms_new = len(atom_indices_frags)

    while no_atoms_old != no_atoms_new:
        no_atoms_old = len(atom_indices_frags)
        for i_ts, ts in enumerate(ts_range):
            for frag_i, frag in enumerate(sim_df['# Atom in Fragment'][ts]):
                for atom_index in atom_indices_frags:
                    int_frag = [ast.literal_eval(i) for i in frag]
                    if atom_index in int_frag and frag_i not in frags_i[i_ts]:
                        frags_i[i_ts].append(frag_i)

        # print(frags_i)

            for i_frag in frags_i[i_ts]:
                #print(i_frag)
                int_frag = [ast.literal_eval(i) for i in sim_df['# Atom in Fragment'][ts][i_frag]]
                #print(int_frag)
                for i_atom in int_frag:
                    if i_atom not in atom_indices_frags:
                        atom_indices_frags.append(i_atom)
                        
        no_atoms_new = len(atom_indices_frags)

    #print(atom_indices_frags)

    atom_indices_frags_new = []
    for i_ts, ts in enumerate(ts_range):

        if i_ts == 0:
            for i_mol in frags_i[i_ts]:
                for smiles in sim_df['SMILES'][ts][i_mol]:
                    #print(smiles)
                    params = rdmolfiles.SmilesParserParams()
                    params.removeHs=False
                    params.sanitize = False
                    mol = rdmolfiles.MolFromSmiles(smiles, params)
                    mol.UpdatePropertyCache(strict=False)
                    rdmolops.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True) # type: ignore

                    for at in mol.GetAtoms():
                        total_charge[i_ts] += at.GetFormalCharge()
                        total_spin[i_ts] += at.GetNumRadicalElectrons()

            #if i_ts == 0:
                frag = sim_df['# Atom in Fragment'][ts][i_mol]
                int_frag = [ast.literal_eval(i) for i in frag]
                atom_indices_frags_new += [at_i for at_i in int_frag]

        string = str("%i\nTIME: %14.7f\n") % (len(atom_indices_frags_new),sim_df['Time step [fs]'][ts])
        f_traj.write(string)
        for i in atom_indices_frags_new:
            string = str("%s %20.10e %20.10e %20.10e\n") % (atom_map[i-1],xyz[ts][i-1][0],xyz[ts][i-1][1],xyz[ts][i-1][2])
            f_traj.write(string)


    #print(total_charge)
    #print(total_spin)

    f_spin_mult.write(str(np.sum(total_charge)) + "\t")
    f_spin_mult.write(str(np.sum(total_spin) + 1))
    
    f_traj.close()
    f_spin_mult.close()