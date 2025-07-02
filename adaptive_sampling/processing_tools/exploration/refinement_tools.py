import ase
from ase.io import read
from ase.units import Bohr, Hartree

import glob
from sella import Sella, Constraints

from pyGSM.utilities import manage_xyz, elements
from pyGSM.coordinate_systems import MyG, Topology

import subprocess

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

def confirm_freq(calculator, mol_file, ):
    return 

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

def de_gsm():
    # Here, we will use the provided gsm binary from pyGSM

    

    return 




