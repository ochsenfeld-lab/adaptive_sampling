import ase
from ase.io import read
from ase.units import Bohr, Hartree

import glob
from sella import Sella

from pyGSM.utilities import manage_xyz, elements
from pyGSM.coordinate_systems import MyG, Topology

import subprocess as sp
import os
import shutil
import re

def sella_opt(calculator: ase.calculators, mol_file: str, order: int, fmax: float=1e-3, nsteps: float=1e3):
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
        internal=True,                    # use internal coordinates
        trajectory=f"optimizer_{mol_file}.traj",  # trajectory file, if exists steps will be appended 
        order=order,                      # 0: Minimum, 1: Transition state
        logfile=f"logfile_{mol_file}",
    )
    opt.run(fmax=fmax, steps=nsteps)
    print('Opt done!')

    return 

def confirm_freq(output_file: str):
    # here the problem is that I have to grep for imaginary and I am not sure how it looks in ORCA
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
    os.makedirs("MIN_FOUND", exist_ok=True)

    for d in os.listdir('.'):
        if not d.startswith('PAT') or not os.path.isdir(d):
            continue

        start_path = os.path.join(d, 'start_opt/freq_analyt/vibfreqs.out')
        end_path = os.path.join(d, 'end_opt/freq_analyt/vibfreqs.out')

        start_count = confirm_freq(start_path)
        end_count = confirm_freq(end_path)

        if start_count == 0 and end_count == 0:
            print(f"Moving {d} to MIN_FOUND/")
            shutil.move(d, os.path.join("MIN_FOUND", d))

def find_viable_rcts():
    
    # Thresholds
    min_threshold = 0.001
    max_threshold = 50.0
    count = 0

    # Source files to copy into ts_opt/
    files_to_copy = [
        "TSnode*",
        "charge_mult*",
    ]

    # Create destination directory if it doesn't exist
    os.makedirs("VIABLE_RCTS", exist_ok=True)

    for d in os.listdir('.'):
        if not d.startswith("PAT") or not os.path.isdir(d):
            continue

        log_path = os.path.join(d, "log")
        if not os.path.isfile(log_path):
            continue

        with open(log_path, 'r') as f:
            log_text = f.read()

        if "Finished GSM!" in log_text and "Ran out of iterations" not in log_text:
            # Extract energy from line like: "TS energy <value>"
            match = re.search(r'TS energy\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', log_text)
            if match:
                try:
                    energy = float(match.group(1))
                except ValueError:
                    continue

                if min_threshold <= energy <= max_threshold:
                    print(f"{d}")
                    print(f"{energy}")
                    print(f"Moving {d} to VIABLE_RCTS")

                    dest_dir = os.path.join("VIABLE_RCTS", d)
                    shutil.move(d, dest_dir)

                    ts_opt_dir = os.path.join(dest_dir, "ts_opt")

                    # Remove and recreate ts_opt/
                    if os.path.exists(ts_opt_dir):
                        shutil.rmtree(ts_opt_dir)
                    os.makedirs(ts_opt_dir)

                    # Copy required files into ts_opt/
                    for pattern in files_to_copy:
                        for file_path in sorted(os.popen(f'ls {pattern} 2>/dev/null').read().splitlines()):
                            if os.path.isfile(file_path):
                                shutil.copy(file_path, ts_opt_dir)

                    count += 1

    print(f"Total matching directories: {count}")


def rearrange_top(raw_input_file: str, gsm_input_file: str):
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
    with open(gsm_input_file, 'w') as f:
        f.writelines(reordered_lines)

    return

def de_gsm(add_args_file: str, out_file:str):
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

    # Here, we will use the provided gsm binary from pyGSM
    #with open(add_args_file, "r") as input:

    #from ase.calculators.orca import OrcaProfile
    #import json
    #profile = OrcaProfile(command='which orca')
    #print(profile)
    #json_dict = json.dumps({"profile": f"{profile}", "orcasimpleinput": "HF STO-3G"})
    #json_dict = json.dumps({"orcasimpleinput": "HF STO-3G"})
    #print(json_dict)
    with open(out_file, "a+") as f:
        sp.run(args=["gsm", "-mode", "DE_GSM", "-charge", f"{charge}", "-multiplicity", f"{mult}"] +  add_args, stdout=f)
    #with open(out_file, "a+") as f:
    #    sp.run(args=["gsm", "-mode", "DE_GSM", "-charge", f"{charge}", "-multiplicity", f"{mult}", "-package", "ase", "--ase-class", "ase.calculators.orca.ORCA", "--ase-kwargs", f"{json_dict}"] +  add_args, stdout=f)

    return

def orca_calc(inp_file: str, temp: float, out_file="orca.out"):
    # here do evth with ORCA in this case
    # call ORCA do to freq analysis on an already opt structure
    with open(out_file, "a+") as f:
        sp.run(args=["orca", inp_file], stdout=f)

    return







