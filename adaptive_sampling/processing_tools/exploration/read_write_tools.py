import pandas as pd
import ast
import json
from typing import Tuple

# read and write  all kinds of file types

def read_traj_file(xyz_file: str) -> Tuple[list, list]:
    ''' Read trajectory file.

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
    '''

    index = -1
    atom_map_total = []
    xyz_list = []
    natoms = 0

    traj_file = open(xyz_file, "r")

    for line_number,line in enumerate(traj_file):
        try:
            vals = line.strip().split()
        except:
            raise Exception ("Failed to parse line...")

        if len(vals) == 1:
            natoms = int(vals[0])
        elif len(vals) == 2:
            timestep = vals[1]
            index += 1
            atom_map_total.append([])
            xyz_list.append([])
        elif len(vals) == 4:
            atom_map_total[index].append(vals[0])
            xyz_list[index].append([float(vals[1]), float(vals[2]), float(vals[3])])

    atom_map = atom_map_total[0]
    traj_file.close()

    return atom_map, xyz_list

def read_bo_file(dat_file: str) -> list:
    ''' Read (Wiberg) bond order (wbo) file.

    Args:
        dat_file: path to bond order file from nanoreactor simulation\n
                  Format: 1 TIME: time step\n
                          2 wbo(0,1)\n
                          3 wbo(0,2)\n
                          4 wbo(0,3)\n
                                .\n
                                .\n
                          only upper triangular (without diagonal elements because equal to 0) is stored to reduce file size\n
    Returns:
        bond_orders = upper half of the bond order matrix stored as list
    '''

    index = -1
    bond_orders = []
    bo_file = open(dat_file,"r")
    for line_number,line in enumerate(bo_file):
        try:
            vals = line.strip().split()
        except:
            raise Exception ("Failed to parse line...")

        if len(vals) == 2:
            timestep = vals[1]
            index += 1
            bond_orders.append([])
        elif len(vals) == 1:
            bond_orders[index].append(float(vals[0]))

    bo_file.close()
    return bond_orders

def read_frag_file(dat_file: str) -> list:
    ''' Read fragment file containing indices belonging to found molecules in each step.

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
    '''

    atom_indices_frag = []
    index = -1

    mols_file = open(dat_file, "r")

    for line in mols_file:
        if line.startswith(' '):
            index += 1
            atom_indices_frag.append([])

        else:
            numbers = line.split()
            atom_indices_frag[index].append(numbers)

    mols_file.close()
    return atom_indices_frag

def write_frag_file(name: str, timestep: float, fragments: list):
    ''' Write fragment file containing indices belonging to found molecules in each step.

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
    '''

    if timestep == 0.0:
        f = open(name,"w")
    else:
        f = open(name,"a")
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
    ''' Restore data frame containing all necessary information from the post-processing from file.

    Args:
        df_file: filepath to .csv file where data frame was stored (sep: ';')
    
    Returns:
        df : pd.DataFrame object with columns 'Time Step', '# Fragment', '# Atom in Fragment', '# Elem in Fragment', 'XYZ', 'SMILES', 'Molecular Formulas'
    '''

    df = pd.read_csv(df_file, sep = ";")

    new_fragments = []
    for ts in range(len(df['# Fragment'])):
        new_fragments.append(ast.literal_eval(df['# Fragment'][ts]))
    
    new_atom_in_fragments = []
    for ts in range(len(df['# Atom in Fragment'])):
        new_atom_in_fragments.append(ast.literal_eval(df['# Atom in Fragment'][ts]))

    new_SMILES = []
    for ts in range(len(df['SMILES'])):
        new_SMILES.append(ast.literal_eval(df['SMILES'][ts]))

    new_molecular_formulas = []
    string = []
    for ts in range(len(df['Molecular Formulas'])):
        string = df['Molecular Formulas'][ts].replace('\'','')
        string = string.replace("[","")
        string = string.replace("]","")
        string = string.replace(" ","")
        new_molecular_formulas.append(string.split(','))     
    
    df['# Fragment'] = new_fragments
    df['# Atom in Fragment'] = new_atom_in_fragments
    df['SMILES'] = new_SMILES
    df['Molecular Formulas'] = new_molecular_formulas
    
    return df

def read_reaction_list(json_file: str) -> list:
    ''' Read JSON file containing reaction data to construct network.

    Args:
        json_file: JSON file path where the reaction list was been stored

    Returns:
        reactions_list: list of reactions\n
                        Format: [event #, [ts_r, ts_p], [smiles_r...], [smiles_p...]]\n
    '''
    with open(json_file, "r") as fp:
        reactions_list = json.load(fp)

    return reactions_list

def get_reaction_traj(traj_file: str, sim_df_file: str, reactions_list_file: str, event_no: list, reaction_traj: str = "rct_traj", all_steps: bool = False):
    
    sim_df = read_trafo_df(sim_df_file)
    atom_map, xyz = read_traj_file(traj_file)
    reactions_list = read_reaction_list(reactions_list_file)
    
    # first, get desired reaction(s) and time steps
    timesteps = []
    reactants = []
    products = []
    atom_indices = []
    
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
                
        atom_indices.extend(reactions_list[event-1][4])
    
    t_start = min(timesteps)
    t_end = max(timesteps)
    

    # get XYZ between the time steps and write file
    f = open(reaction_traj + "_" + str(event_no[0]) + "_" + str(event_no[-1]) + ".xyz", "a+")
    
    if not all_steps:
        for ts in [t_start, t_end]:
            string = str("%i\nTIME: %14.7f\n") % (len(atom_indices),sim_df['Time step [fs]'][ts])
            f.write(string)
            for i in atom_indices:
                i -= 1
                string = str("%s %20.10e %20.10e %20.10e\n") % (atom_map[i],xyz[ts][i][0],xyz[ts][i][1],xyz[ts][i][2])
                f.write(string)
    else:       
        # This variant is for printing all steps in the trajectory
        for ts in range(min(timesteps),max(timesteps)):
            string = str("%i\nTIME: %14.7f\n") % (len(atom_indices),sim_df['Time step [fs]'][ts])
            f.write(string)
            for i in atom_indices:
                i -= 1
                string = str("%s %20.10e %20.10e %20.10e\n") % (atom_map[i],xyz[ts][i][0],xyz[ts][i][1],xyz[ts][i][2])
                f.write(string)
    
    f.close()

    return 