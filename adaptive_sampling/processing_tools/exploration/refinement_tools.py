import ase
from ase.io import read, write
from ase.units import Bohr, Hartree

import glob
from sella import Sella

import pyGSM
from pyGSM.utilities import manage_xyz, elements
from pyGSM.coordinate_systems import MyG, Topology

import networkx as nx

import subprocess as sp
import os
import shutil
import re
from pathlib import Path

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
        trajectory=f"{mol_file}.traj", # trajectory file, if exists steps will be appended 
        order=order,                   # 0: Minimum, 1: Transition state
        logfile=f"opt_logfile.out",    # optimization log file
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
    output_file: str
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
                if 'imaginary' in line.lower():
                    match = re.search(r'imaginary perturbations\s+\.+\s+(\d+)', line)
                    if match:
                        return int(match.group(1))
    except FileNotFoundError:
        pass
    return 

def find_minima():
    """Identify directories where both start and end structures are minima (no imaginary frequencies).
    Moves such directories to a new directory named 'MIN_FOUND'.
    """
    os.makedirs("MIN_FOUND", exist_ok=True)

    for d in os.listdir('.'):
        print(d)
        if not d.startswith('pattern') or not os.path.isdir(d):
            continue

        start_path = os.path.join(d, 'start_opt/orca.out')
        end_path = os.path.join(d, 'end_opt/orca.out')

        start_count = confirm_freq(start_path)
        end_count = confirm_freq(end_path)

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
                                shutil.copy(file_path, f"ts_opt/{file_path.name}")

                    os.chdir("../..")
                    #print(os.getcwd())
                    count += 1

    print(f"Total matching directories: {count}")
    os.chdir("VIABLE_RCTS")

    return











