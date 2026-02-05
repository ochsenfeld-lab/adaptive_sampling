import ase
from ase.io import read, write

import glob
try:
    from sella import Sella
except (ImportError, RuntimeError) as e:
    print(f"Sella not imported due to incompatibilities: {e}")
import os
from collections import Counter

from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdDetermineBonds
import rdkit.Chem.rdmolops as rdmolops

import json

from pyGSM.utilities import manage_xyz, elements
from pyGSM.coordinate_systems import MyG, Topology

import networkx as nx

import subprocess as sp
import shutil
import re
from pathlib import Path
from typing import Tuple

HARTREE_TO_KCAL_MOL = 627.503
number_pattern = re.compile(r"[-+]?\d*\.\d+|\d+")

# Opt/Calc functions

def sella_opt(
    calculator: ase.calculators,
    mol_file: str,
    order: int, fmax: float=1e-3,
    nsteps: float=1e3
) -> None:
    """Perform a geometry optimization using the Sella optimizer.
    
    Args:
        calculator (ase.calculators): An ASE calculator object for energy and force evaluations.
        mol_file (str): Path to the input molecular structure file (e.g., XYZ format).
        order (int): Optimization order (0 for minimum, 1 for transition state).
        fmax (float, optional): Maximum force convergence criterion. Defaults to 1e-3.
        nsteps (int, optional): Maximum number of optimization steps. Defaults to 1000.
        
    Returns:
        None. The optimized structure is saved to a .traj file.
    """
    charge = 0
    mult = 1
    # Read charge and multiplicity from the extracted pattern
    pattern = "charge_mult*"
    for filename in glob.glob(pattern):
        with open(filename, "r") as f:
            for line in f:
                charge, mult = map(int, line.strip().split("\t"))

    mol = read(mol_file)
    mol.calc = calculator

    opt = Sella(
        mol,
        internal=True,                 # use internal coordinates
        trajectory=f"opt_{mol_file}.traj", # trajectory file, if exists steps will be appended 
        order=order,                   # 0: Minimum, 1: Transition state
        logfile=None#f"opt_logfile.out",    # optimization log file
    )
    opt.run(fmax=fmax, steps=nsteps)
    print('Opt done!')

    return 

def ase_calc(
    calculator: ase.calculators, 
    mol_file: str
) -> None:
    """Perform a single-point energy calculation using the specified ASE calculator.
    This can be also be an optimization if the calculator is set up that way.
    
    Args:
        calculator (ase.calculators): An ASE calculator object for energy and force evaluations.
        mol_file (str): Path to the input molecular structure file (e.g., XYZ format).
        
    Returns:
        None. The energy is computed and stored in the calculator object and the calculation is
        logged by the external program.
    """
    mol = read(mol_file)
    mol.calc = calculator

    mol.get_potential_energy()

    return

def orca_calc(
    inp_file: str, 
    out_file="orca.out"
) -> None:
    """Perform any type of calculation using ORCA at command line level.
    The function runs ORCA with the specified input file and appends the output to the given output file.
    
    Args:
        inp_file (str): Path to the ORCA input file (.inp).
        out_file (str, optional): Path to the output file where ORCA output will be saved. Defaults to "orca.out".
    
    Returns:
        None. The ORCA output is saved to the specified output file.
    """
    # here do evth with ORCA in this case
    # call ORCA do to freq analysis on an already opt structure
    with open(out_file, "a+") as f:
        sp.run(args=["orca", inp_file], stdout=f)

def de_gsm(
    add_args_file: str, 
    out_file:str, 
    index: int=0
) -> None:
    """Run the DE-GSM method using the pyGSM package with specified additional arguments.
    The function sets up the environment, reads charge and multiplicity, and executes the GSM command
    with the provided arguments. Temporary directories are cleaned up after execution.

    Args:
        add_args_file (str): Path to a file containing additional command-line arguments for GSM.
        out_file (str): Path to the output file where GSM output will be saved.
        index (int, optional): An index used for temporary directory naming. Defaults to 0.
        
    Returns:
        None. The GSM output is saved to the specified output file.
    """
    charge = 0
    mult = 1
    # Read charge and multiplicity from the extracted pattern
    pattern = "charge_mult*"
    for filename in glob.glob(pattern):
        with open(filename, "r") as f:
            for line in f:
                charge, mult = map(int, line.strip().split("\t"))
            f.close()

    print(f"Charge: {charge}")
    print(f"Multiplicity: {mult}")

    import shlex
    with open(add_args_file, 'r') as f:
        add_args = shlex.split(f.read())
        f.close()
    # Here, we will use the provided gsm binary from pyGSM
    os.environ["PBS_JOBID"] = str(index)

    with open(out_file, "a+") as f:
        sp.run(args=["gsm", "-xyzfile", "reordered_init_geom.xyz", "-mode", "DE_GSM", "-charge", f"{charge}", "-multiplicity", f"{mult}"] +  add_args, stdout=f)

    shutil.rmtree(f"/tmp/{index}")
    shutil.rmtree("scratch")

    return

# functions handling geometry files

def convert_geom_rm_X(
    input_file: str
) -> None:
    """Convert a molecular geometry file by removing atoms labeled 'X' and updating the atom count.
    This is sometimes necessary for compatibility with certain software.

    Args:
        input_file (str): Path to the input molecular geometry file (e.g., XYZ format).

    Returns:
        None. The modified geometry is saved to a new file with 'X' atoms removed.
    """
    mol = read(input_file, index=":") 
    write(f"{input_file[:-5]}", mol[-1], format='xyz')
    natoms = 0

    with open(f"{input_file[:-5]}", "r") as xyzr:
        with open('tmp.txt', "w") as xyzw:
            lines = iter(xyzr)
            for i, line in enumerate(lines):
                words = line.strip().split()
                if i == 0:
                    xyzw.write(line)
                elif len(words) == 0:
                    xyzw.write(line)
                elif words[0] != 'X':
                    natoms += 1
                    xyzw.write(line)

    with open('tmp.txt', "r") as xyzr:
        with open('tmp2.txt', "w") as xyzw:
            lines = iter(xyzr)
            for i, line in enumerate(lines):
                if i == 0:
                    xyzw.write(f"{natoms}\n")
                else:
                    xyzw.write(line)

    os.replace('tmp2.txt', f"{input_file[:-5]}")

def rearrange_top(
    raw_input_file: str
) -> None:
    """Rearrange the atoms in a molecular geometry file based on connectivity.
    The function reads the input file, builds a molecular topology, identifies connected components,
    and reorders the atoms accordingly. The reordered geometry is saved to a new file. This is needed 
    for the DE-GSM method to work properly.

    Args:
        raw_input_file (str): Path to the input molecular geometry file (e.g., XYZ format).
        
    Returns:
        None. The reordered geometry is saved to 'reordered_init_geom.xyz'.
    """
    geoms = manage_xyz.read_xyzs(raw_input_file)

    # Build the topology
    atom_symbols = manage_xyz.get_atoms(geoms[0])
    ELEMENT_TABLE = elements.ElementData()
    atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atom_symbols]
    natoms = len(atom_symbols)
    xyz1 = manage_xyz.xyz_to_np(geoms[0])
    top1 = Topology.build_topology(
        xyz1,
        atoms,
    )

    #print(top1.edges())

    xyz2 = manage_xyz.xyz_to_np(geoms[-1])
    top2 = Topology.build_topology(
        xyz2,
        atoms,
    )

    for bond in top2.edges():
        if bond in top1.edges:
            pass
        elif (bond[1], bond[0]) in top1.edges():
            pass
        else:
            if bond[0] > bond[1]:
                top1.add_edge(bond[0], bond[1])
            else:
                top1.add_edge(bond[1], bond[0])

    order_list_e = [0, 1] # for the first two line
    order_list_p = [natoms + 2, natoms + 3]
    with open("frags.txt", "a+") as file:
        fragments_graphs = [top1.subgraph(c).copy() for c in nx.connected_components(top1)]
        for g in fragments_graphs:
            g.__class__ = MyG
            for index in g.L():
                order_list_e.append(index + 2)
                order_list_p.append(index + natoms + 4)
                file.write(str(index) + " ")

            file.write("\n")
    file.close()

    new_order = order_list_e + order_list_p

    with open(raw_input_file, 'r') as f:
        lines = f.readlines()

    # Ensure indices are within bounds
    if max(new_order) >= len(lines) or min(new_order) < 0:
        raise ValueError("Index out of range")

    # Reorder lines
    reordered_lines = [lines[i] for i in new_order]

    # Write reordered lines to the output file
    with open(f"reordered_init_geom.xyz", 'w') as f:
        f.writelines(reordered_lines)

    return

def confirm_freq(
    output_file: str,
    search_term: str
) -> int:
    """Check the output file for imaginary frequencies.
    
    Args:
        output_file (str): Path to the output file containing frequency analysis results.
        
    Returns:
        int: Number of imaginary frequencies found. Returns None if none are found or file is missing.
    """
    try:
        with open(output_file, 'r') as f:
            for line in f:
                if search_term in line.lower():
                    # Try number before term
                    match = re.search(rf'(\d+)\s+{re.escape(search_term)}', line)
                    if match:
                        return int(match.group(1))
                    # Try term then dots then number
                    match = re.search(rf'{re.escape(search_term)}\s+\.+\s+(\d+)', line)
                    if match:
                        return int(match.group(1))
    except FileNotFoundError:
        pass
    return 

def find_minima(start_output: str='start_opt/orca.out',
                end_output: str='end_opt/orca.out',
                search_term: str='imaginary perturbations'
                ) -> None:
    """Identify directories where both start and end structures are minima (no imaginary frequencies).
    Moves such directories to a new directory named 'MIN_FOUND'.
    """
    os.makedirs("MIN_FOUND", exist_ok=True)

    for d in os.listdir('.'):
        if not d.startswith('pattern') or not os.path.isdir(d):
            continue

        print(d)
        start_path = os.path.join(d, start_output)
        end_path = os.path.join(d, end_output)

        start_count = confirm_freq(start_path, search_term)
        end_count = confirm_freq(end_path, search_term)

        if start_count == 0 and end_count == 0:
            print(f"Moving {d} to MIN_FOUND/")
            shutil.move(d, os.path.join("MIN_FOUND", d))

def find_viable_rcts(
    min_threshold: float=0.001, 
    max_threshold: float=50.0
) -> None:
    """Identify directories with successful GSM runs and TS energies within specified thresholds.
    Moves such directories to a new directory named 'VIABLE_RCTS' and organizes necessary files for TS optimization. 
    
    Args:
        min_threshold (float, optional): Minimum energy threshold. Defaults to 0.001 kcal/mol.
        max_threshold (float, optional): Maximum energy threshold. Defaults to 50.0 kcal/mol.
        
    Returns:
        None. The identified directories are moved and organized.
    """    
    # Initialize counter for matching directories
    count = 0

    # Source files to copy into ts_opt/
    files_to_copy = [
        "TSnode*",
        "charge_mult*",
    ]

    # Create destination directory if it doesn't exist
    os.makedirs("VIABLE_RCTS", exist_ok=True)

    for d in os.listdir('.'):
        if not d.startswith("pattern") or not os.path.isdir(d):
            continue

        log_path = os.path.join(d, "log")
        print(log_path)
        if not os.path.isfile(log_path):
            continue

        with open(log_path, 'r') as f:
            log_text = f.read()

        if "Finished GSM!" in log_text and "Ran out of iterations" not in log_text:
            # Extract energy from line like: "TS energy <value>"
            match = re.search(r' TS energy:\s+([0-9.]+)', log_text)
            #print(match)
            if match:
                try:
                    energy = float(match.group(1))
                    #print(energy)
                except ValueError:
                    continue

                if min_threshold <= energy <= max_threshold:
                    print(f"{d}")
                    print(f"{energy}")
                    print(f"Moving {d} to VIABLE_RCTS")

                    dest_dir = os.path.join("VIABLE_RCTS", d)
                    shutil.move(d, dest_dir)

                    ts_opt_dir = os.path.join(dest_dir, "ts_opt")

                    ts_opt_dir = Path(ts_opt_dir)
                    ts_opt_dir.mkdir(parents=True, exist_ok=True)

                    os.chdir(dest_dir)
                    print(os.getcwd())
                    for pattern in files_to_copy:              
                        for file_path in sorted(Path().glob(pattern)):
                            if file_path.is_file():
                                if file_path.name.startswith("TSnode"):
                                    shutil.copy(file_path, f"ts_opt/ts.xyz")
                                else:
                                    shutil.copy(file_path, f"ts_opt/{file_path.name}")

                    os.chdir("../..")
                    #print(os.getcwd())
                    count += 1

    print(f"Total matching directories: {count}")
    os.chdir("VIABLE_RCTS")

    return

def find_successful_reactions(search_term_freq: str='imaginary perturbations',
                              search_term_gibbs: str='Final Gibbs free energy',
                              out_file: str="orca.out") -> None:
    """Identify directories with successful GSM runs and confirmed stationary points and move them to the 'SUCCESSFUL_RCTS' directory.
    """
    os.makedirs("SUCCESSFUL_RCTS", exist_ok=True)
    for d in os.listdir('.'):
        if not d.startswith('pattern') or not os.path.isdir(d):
            continue

        ts_path = os.path.join(d, f'ts_opt/{out_file}')

        ts_count = confirm_freq(ts_path, search_term_freq)

        if ts_count == 1:
            print(f"Moving {d} to SUCCESSFUL_RCTS/")
            shutil.move(d, os.path.join("SUCCESSFUL_RCTS", d))

    generate_refined_network(directory="SUCCESSFUL_RCTS", search_term=search_term_gibbs, out_file=out_file)

    return

def write_file(
    content: str, 
    filepath: str
) -> None:
    """Write content to a specified file, creating directories as needed.
    
    Args:
        content (str): The content to write to the file.
        filepath (str): The path to the file where the content should be written.
        
    Returns:
        None. The content is written to the specified file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"File written to {path}")

def generate_refined_network(
    directory: str,
    prefixes: tuple=("start", "end", "ts"),
    search_term: str="Final Gibbs free energy",
    out_file: str="orca.out"
) -> None:
    """Generate a refined reaction network from successful reactions.
    This function processes the 'SUCCESSFUL_RCTS' directory to create a refined reaction network.
    
    Args:
        directory (str): The path to the directory containing successful reactions.
        
    Returns:
        None. The refined reaction network is generated and saved in a JSON format.
    """
    print(f"Generating refined reaction network for {directory}")
    reactions_list = extract_refined_reactions(root_dir=directory, prefixes=prefixes, 
                                               search_term=search_term, out_file=out_file)

    with open("reactions_list_refined.json", "w+") as f:
        json.dump(reactions_list, f)
    f.close()

    return

# Open the file and read line by line
def determine_charge(
    dirname: str, 
    pattern="charge_mult*"
) -> Tuple[int, int]:
    """Determine the charge and multiplicity from a specified file in a directory.
    
    Args:
        dirname (str): The directory to search for the file.
        pattern (str, optional): The filename pattern to search for. Defaults to "charge_mult*".
        
    Returns:
        Tuple[int, int]: A tuple containing the charge and multiplicity.
    """
    for filename in dirname.glob(pattern):
        with open(filename, "r") as f:
            for line in f:
                charge, mult = map(int, line.strip().split("\t"))  # Split by tab and convert to integers
    return charge, mult

def extract_refined_reactions(root_dir: Path="SUCCESSFUL_RCTS", prefixes: tuple=("start", "end", "ts"), 
                              search_term: str="Final Gibbs free energy", out_file: str="orca.out") -> list:
    root_dir = Path(root_dir).resolve()
    reactions_list = []
    event_counter = 1
    for d in root_dir.glob("pattern*"):
        print(d)
        energies = []
        reaction = []
        for f in d.glob("pattern*"):
            reaction.append(f"Event {event_counter}")
            reaction.append(get_reaction_time(f))
        transition_state = []
        for prefix in prefixes:
            prefix_path = d / f"{prefix}_opt" / out_file
            if prefix_path.exists():  # Check if the file exists
                with prefix_path.open() as file:
                    for line in file:
                        if search_term in line:
                            match = number_pattern.findall(line)
                            if len(match) == 1:
                                extracted_number = float(match[0])  # Convert to float
                                energies.append(extracted_number)

            charge, mult = determine_charge(d)
            smiles = xyz2mol(d / f"{prefix}_opt" / f"opt_{prefix}.xyz", charge=charge)
            if "ts" not in prefix:
                reaction.append(smiles.split("."))
            else:
                transition_state = smiles.split(".")
                for i,smiles in enumerate(transition_state):
                    transition_state[i] = removeAtomMap(smiles)

        count1, count2 = Counter(reaction[2]), Counter(reaction[3])

        # Determine the number of times to remove each common element
        if count1 != count2:
            cat = count1 & count2  # This gets the minimum count directly
            # Create filtered lists
            reaction[2] = remove_common_occurrences(reaction[2], cat.copy())  # Use copy to avoid modifying original
            reaction[3] = remove_common_occurrences(reaction[3], cat.copy())
        else:
            cat = Counter()

        # remove atom maps from SMILES as we have already determined the catalysts
        for i, smiles in enumerate(reaction[2]):
            reaction[2][i] = removeAtomMap(smiles)

        for i, smiles in enumerate(reaction[3]):
            reaction[3][i] = removeAtomMap(smiles)

        new_cat=[]
        for smiles in cat:
            new_smiles = removeAtomMap(smiles)
            new_cat.append(new_smiles)

        cat = Counter(new_cat)

        # add important parameters to the dictionary
        reaction_free_energy = (energies[1] - energies[0]) * HARTREE_TO_KCAL_MOL
        activation_free_energy = (energies[-1] - energies[0]) * HARTREE_TO_KCAL_MOL
        reaction.append({"rxn_free_energy": reaction_free_energy, "rxn_barrier": activation_free_energy, "ts": transition_state, "cat": cat, "energies": energies, "dir": str(d)})
        reaction[0] = "".join(reaction[0])
        reactions_list.append(reaction)
        event_counter += 1

    return reactions_list

def get_reaction_time(
    raw_rct_list: str
) -> list:
    """Extract reaction times from a specified file.
    
    Args:
        pattern_file (str): Path to the file containing reaction time information.
        
    Returns:
        list: A list of reaction times extracted from the file.
    """
    ts = []
    if raw_rct_list.exists():  # Check if the file exists
        with raw_rct_list.open() as file:
            for line in file:
                if "TIME:" in line:
                    match = number_pattern.findall(line)
                    ts.append(int(float(match[0]) / 0.5 / 50.0))
    return ts

def xyz2mol(
    xyz_file: str=None,
    charge: int = 0
) -> str:
    """Convert an XYZ file to a SMILES string using RDKit.
    
    Args:
        xyz_file (str): Path to the XYZ file.
        charge (int, optional): The charge of the molecule. Defaults to 0.
        
    Returns:
        str: The SMILES representation of the molecule.
    """
    mol = rdmolfiles.MolFromXYZFile(str(xyz_file))
    conn_mol = Chem.Mol(mol)
    rdDetermineBonds.DetermineBonds(conn_mol, charge=charge, useAtomMap=True)
    smiles = Chem.MolToSmiles(conn_mol)
    return smiles

def removeAtomMap(
    smiles: str
) -> str:
    """Remove atom mapping numbers from a SMILES string.
    
    Args:
        smiles (str): The input SMILES string with atom mapping numbers.
        
    Returns:
        str: The SMILES string without atom mapping numbers.
    """
    # transform smiles to mol again and back for uniformization and avoid adjustHs
    params = rdmolfiles.SmilesParserParams()
    params.removeHs=False
    params.sanitize = False
    mol = rdmolfiles.MolFromSmiles(smiles, params)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    mol.UpdatePropertyCache(strict=False)
    rdmolops.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS, catchErrors=True)
    smiles = rdmolfiles.MolToSmiles(mol)
    return smiles

def remove_common_occurrences(
    lst: list, 
    common_counts: Counter
) -> list:
    """Remove common occurrences from a list based on a Counter of common elements.
    Args:
        lst (list): The input list from which to remove common occurrences.
        common_counts (Counter): A Counter object containing elements to remove and their counts.

    Returns:
        list: A new list with common occurrences removed.
    """
    new_list = []
    for item in lst:
        if common_counts[item] > 0:
            common_counts[item] -= 1
        else:
            new_list.append(item)
    return new_list











