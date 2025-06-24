import sys
import pandas as pd
import numpy as np
from rdkit.Chem import rdmolops  
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from .utils import *
from .read_write_utils import *

# dictionary for standard experimental bond lengths (in Angstrom) defined globally to be used in classes and functions
global __std_bond_lengths__
__std_bond_lengths__ = {'H':  {'H': 0.741, 'Li': 1.595, 'C': 1.120, 'N':  1.036, 'O':  0.970, 'F':  0.917, 'P': 1.422, 'S': 1.341},
                        'Li': {'H': 1.595, 'Li': 2.673, 'C': np.NaN, 'N': np.NaN, 'O': 1.688, "F": 1.564, 'P': np.NaN, 'S': 2.150},
                        'C':  {'H': 1.120, 'Li': np.NaN, 'C': 1.243, 'N': 1.172, 'O': 1.128, 'F': 1.276, 'P': 1.562, 'S': 1.535},
                        'N':  {'H': 1.036, 'Li': np.NaN, 'C': 1.172, 'N': 1.098, 'O': 1.154, 'F': 1.317, 'P': 1.491, 'S': 1.497},
                        'O':  {'H': 0.970, 'Li': 1.688, 'C': 1.128, 'N': 1.154, 'O': 1.208, 'F': 1.354, 'P': 1.476, 'S': 1.481},
                        'F':  {'H': 0.917, 'Li': 1.564, 'C': 1.276, 'N': 1.317, 'O': 1.354, 'F': 1.412, 'P': 1.593, 'S': 1.599},
                        'P':  {'H': 1.422, 'Li': np.NaN, 'C': 1.562, 'N': 1.491, 'O': 1.476, 'F': 1.593, 'P': 1.893, 'S': 1.900},
                        'S':  {'H': 1.345, 'Li': 2.150,'C': 1.535, 'N': 1.497, 'O': 1.481, 'F': 1.599, 'P': 1.900, 'S': 1.889}
                        }

class NanoSim:
    ''' Class for Nanoreactor Simulation Results
        
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
    '''
    def __init__(self, traj_file: str, bond_order_file: str, mols_file: str = None, dt: float = 0.5, read_rate: int = 50):
        self.dt = dt
        self.read_rate = read_rate

        self.atom_map, self.traj = read_traj_file(traj_file)
        self.bond_orders = read_bo_file(bond_order_file)
        if mols_file != None:
            self.fragments = read_frag_file(mols_file)
        else:
            self.fragments = []
        
        # Initialize other variables
        self.elem_fragments = []
        self.smiles = []
        self.mol_formulas = []
        self.df = pd.DataFrame()

    def calc_frags(self, ts: int, bond_order_matrix: np.ndarray, buffer_type: str = "Ar"):
        ''' Group atoms into fragments based on atom indices for given time step. Bonds are considered above 0.5 for the (Wiberg) bond orders.
            If the WBO's of one atom do not exceed 0.5 then a second criterion based on relative bond lengths is employed.
            Buffer type needs to be specified, the default is argon.

        Args:
            ts: time step index
            bond_order_matrix: Numpy array containing complete bond order matrix for ts
            buffer_type: element used as buffer; default is argon
        Returns:
            fragments: list of lists containing the fragments based on atom indices for ts
        '''

        global __std_bond_lengths__

        if len(bond_order_matrix) == 0:
            print ("WBO required, when calculating fragments!")
            sys.exit(1)

        fragments = []
        natoms = len(self.atom_map)

        for i in range(0,natoms):
            for j in range(i+1,natoms):
                if bond_order_matrix[i][j] < 0.5e0: continue # skip if no bond
                at1 = i + 1
                at2 = j + 1
                found = []
                count = 0

                for fragment in fragments:
                    if is_in(at1,fragment) and is_in(at2,fragment):
                        found.append(count)
                    elif is_in(at1,fragment):
                        fragment.append(at2)
                        found.append(count)
                    elif is_in(at2,fragment):
                        fragment.append(at1)
                        found.append(count)
                    count += 1

                if len(found) == 0:
                    fragments.append([at1,at2])
                elif len(found) > 1:
                    new_fragments = []
                    nfrag = len(found)
                    f  = []
                    for nfrag_i in range(nfrag):
                        f.append(fragments[found[nfrag_i]])

                    fscr = f[0]
                    for nfrag_i in range(1,nfrag):
                        fscr = fscr + list(set(f[nfrag_i]) - set(fscr))

                    new_fragments.append(fscr)
                    for nfrag_i in range(len(fragments)):
                        if is_in(nfrag_i,found) == 0:
                            new_fragments.append(fragments[nfrag_i])

                    fragments = []
                    fragments = new_fragments

        for atom1 in range(1,natoms+1):
            atom1_type = self.atom_map[atom1-1].capitalize()

            distance_array = np.empty(natoms)
            distance_array[:] = np.NaN

            atomic_type_list = []

            sd_array = np.empty(natoms)
            sd_array[:] = np.NaN

            bool_check_atom1 = any(atom1 in sublist for sublist in fragments)
            if (bool_check_atom1): continue
            elif atom1_type == buffer_type:
                fragments.append([atom1])
                continue

            for atom2 in range(1, natoms+1):
                atom2_type = self.atom_map[atom2-1].capitalize()
                bool_check_atom2 = any(atom2 in sublist for sublist in fragments)
                if atom1==atom2:
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
                    coord_atom1 = np.array(self.traj[ts][atom1-1])
                    coord_atom2 = np.array(self.traj[ts][atom2-1])
                    distance = np.linalg.norm(coord_atom1-coord_atom2)

                    distance_array[atom2-1] = distance
                    distance_array[atom1-1] = np.NaN

                    sd_array[atom2-1] = np.sqrt(pow(distance_array[atom2-1] - __std_bond_lengths__[atom1_type][atom2_type],2))

            atom_partner = np.nanargmin(sd_array) + 1

            bool_check_atom_partner = any(atom_partner in sublist for sublist in fragments)

            if bool_check_atom_partner:
                for frag in fragments:
                    if atom_partner in frag:
                        frag.append(atom1)
            else:
                fragments.append([atom1,atom_partner])

        return fragments

    def generate_frag_lists(self):
        ''' Compute fragments for all time steps and write file containing fragment information.
            This can also be done on-the-fly during the MD simulation.

        Args:
            -
        Returns:
            -
        '''

        natoms = len(self.atom_map)
        fragments = []
        
        for ts in range(len(self.bond_orders)):
            bond_order_ts = self.bond_orders[ts]
            bond_order_matrix = np.zeros((natoms,natoms))

            for i in range(natoms):
                for j in range(i+1,natoms):

                    n = natoms

                    bond_order_matrix[i][i] = 0

                    index = int((n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1)

                    bond_order_matrix[i][j] = bond_order_ts[index]
                    bond_order_matrix[j][i] = bond_order_matrix[i][j]

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

    def mol_from_graphs(self, node_list: list, adjacency_matrix: list, radicals: bool) -> Chem.Mol():
        ''' Compute RDKit mol object from atom number list and adjacency matrix (WBO matrix in this case).

        Args:
            node_list: list of atomic numbers
            adjacency_matrix: WBO matrix for fragment of interest
        Returns:
            mol: RDKit mol object
        '''

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            a = Chem.Atom(node_list[i])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        # add bonds between adjacent atoms
        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):

                # only traverse half the matrix
                if iy <= ix:
                    continue

                # add relevant bond type (there are many more of these)
                if bond == 0:
                    continue
                elif bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)


        # Convert RWMol to Mol object
        mol = mol.GetMol()
        if radicals:
            for atom in mol.GetAtoms():
                atom.SetNoImplicit(True)
            rdmolops.AssignRadicals(mol)
        else:
            mol.UpdatePropertyCache(strict=False)
            no_atoms_mol = mol.GetNumAtoms()
            for at_index in range(no_atoms_mol):
                at = mol.GetAtomWithIdx(at_index)
                z_atom = at.GetAtomicNum()
                # H atoms
                if at.GetAtomicNum()==1 and at.GetFormalCharge()==0 and at.GetExplicitValence()==0:
                    at.SetFormalCharge(+1)
                # Li atoms
                if at.GetAtomicNum()==3 and at.GetFormalCharge()==0 and at.GetExplicitValence()==0:
                    at.SetFormalCharge(+1)
                if at.GetAtomicNum()==3 and at.GetFormalCharge()==0 and at.GetExplicitValence()==1:
                    at.SetFormalCharge(-1)
                # C atoms
                if at.GetAtomicNum()==6 and at.GetFormalCharge()==0 and at.GetExplicitValence()==0:
                    at.SetFormalCharge(-4)
                if at.GetAtomicNum()==6 and at.GetFormalCharge()==0 and at.GetExplicitValence()==1:
                    at.SetFormalCharge(-3)
                if at.GetAtomicNum()==6 and at.GetFormalCharge()==0 and at.GetExplicitValence()==2:
                    at.SetFormalCharge(-2)
                if at.GetAtomicNum()==6 and at.GetFormalCharge()==0 and at.GetExplicitValence()==3:
                    at.SetFormalCharge(-1)
                if at.GetAtomicNum()==6 and at.GetFormalCharge()==0 and at.GetExplicitValence()==5:
                    at.SetFormalCharge(+1)
                # N atoms
                if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==0:
                    at.SetFormalCharge(-3)
                if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==1:
                    at.SetFormalCharge(-2)
                if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==2:
                    at.SetFormalCharge(-1)
                if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                    at.SetFormalCharge(+1)
                # O atoms
                if at.GetAtomicNum()==8 and at.GetFormalCharge()==0 and at.GetExplicitValence()==0:
                    at.SetFormalCharge(-2)
                if at.GetAtomicNum()==8 and at.GetFormalCharge()==0 and at.GetExplicitValence()==1:
                    at.SetFormalCharge(-1)
                if at.GetAtomicNum()==8 and at.GetFormalCharge()==0 and at.GetExplicitValence()==3:
                    at.SetFormalCharge(+1)
                #F atoms
                if at.GetAtomicNum()==9 and at.GetFormalCharge()==0 and at.GetExplicitValence()==0:
                    at.SetFormalCharge(-1)
                if at.GetAtomicNum()==9 and at.GetFormalCharge()==0 and at.GetExplicitValence()==2:
                    at.SetFormalCharge(+1)
                # P atoms
                if at.GetAtomicNum()==15 and at.GetFormalCharge()==0 and at.GetExplicitValence()==0:
                    at.SetFormalCharge(-3)
                if at.GetAtomicNum()==15 and at.GetFormalCharge()==0 and at.GetExplicitValence()==1:
                    at.SetFormalCharge(-2)
                if at.GetAtomicNum()==15 and at.GetFormalCharge()==0 and at.GetExplicitValence()==2:
                    at.SetFormalCharge(-1)
                if at.GetAtomicNum()==15 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                    at.SetFormalCharge(+1)
                if at.GetAtomicNum()==15 and at.GetFormalCharge()==0 and at.GetExplicitValence()==6:
                    at.SetFormalCharge(-1)
                # S atoms                                                                                                                                    
                if at.GetAtomicNum()==16 and at.GetFormalCharge()==0 and at.GetExplicitValence()==0:                                                    
                    at.SetFormalCharge(-2)                                                                                                               
                if at.GetAtomicNum()==16 and at.GetFormalCharge()==0 and at.GetExplicitValence()==1:  
                    at.SetFormalCharge(-1)                                                                                                                
                if at.GetAtomicNum()==16 and at.GetFormalCharge()==0 and at.GetExplicitValence()==3:
                    at.SetFormalCharge(+1)                                                                                                                
                if at.GetAtomicNum()==16 and at.GetFormalCharge()==0 and at.GetExplicitValence()==5:                                                  
                    at.SetFormalCharge(+1)                                                                                                                   

        # Recalculate valences and sanitize mol
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|
                        Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
                        Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                        catchErrors=True)
        return mol

    def generate_mols_smiles(self, radicals: bool = False):
        ''' Compute SMILES and corresponding molecular formulas from lists of fragments and bond order matrix.

        Args:
            -
        Returns:
            -
        '''

        for ts in range(len(self.fragments)):
            self.elem_fragments.append([])
            for i_frag, frag in enumerate(self.fragments[ts]):
                self.elem_fragments[ts].append([])
                for i in frag:
                    self.elem_fragments[ts][i_frag].append(self.atom_map[int(i)-1])
        
        natoms = len(self.atom_map)
        mols = []
        
        for ts in range(len(self.fragments)):
            bond_order_ts = self.bond_orders[ts]
            bond_order_matrix = np.zeros((natoms,natoms))
            self.smiles.append([])
            self.mol_formulas.append([])
            mols.append([])

            for i in range(natoms):
                for j in range(i+1,natoms):

                    n = natoms

                    bond_order_matrix[i][i] = 0

                    index = int((n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1)

                    bond_order_matrix[i][j] = round_bond_order(bond_order_ts[index])
                    bond_order_matrix[j][i] = bond_order_matrix[i][j]

            # Go through bond orders list and write for every fragment the correct bond order matrix and then convert it to SMILES

            for fragment_index in range(len(self.fragments[ts])):
                bond_order_matrix_fragment = np.zeros((len(self.fragments[ts][fragment_index]),len(self.fragments[ts][fragment_index])))

                for i in range(len(self.fragments[ts][fragment_index])):
                    atom_index1 = int(self.fragments[ts][fragment_index][i])-1

                    for j in range(i+1,len(self.fragments[ts][fragment_index])):
                        atom_index2 = int(self.fragments[ts][fragment_index][j])-1

                        bond_order_matrix_fragment[i][i] = 0
                        bond_order_matrix_fragment[i][j] = bond_order_matrix[atom_index1][atom_index2]
                        bond_order_matrix_fragment[j][i] = bond_order_matrix_fragment[i][j]

                atoms = [int_atom(atom) for atom in self.elem_fragments[ts][fragment_index]]

                try:
                    # get mol and smiles
                    mol = self.mol_from_graphs(atoms, bond_order_matrix_fragment, radicals)
                    smile = Chem.MolToSmiles(mol)
                    # transform smiles to mol again and back for uniformization and avoid adjustHs
                    mol = Chem.MolFromSmiles(smile,sanitize=False)
                    Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|
                                     Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
                                     Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                                     catchErrors=True)
                    smile = Chem.MolToSmiles(mol)
                    # calculate formula and add it to df
                    formula = CalcMolFormula(mol)
                    self.mol_formulas[ts].append(formula)
                except:
                    print("Revise structure")
                    mol = ["Revise structure"]
                    smile = "Revise structure"

                mols[ts].append(mol)
                self.smiles[ts].append(smile.split("."))
        
        return 

    def generate_df(self):
        ''' Generate data frame after the evaluation of a nanoreactor simulation is finished.

        Args:
            -
        Returns:
            - 
        '''

        column_names = ['Time step [fs]', '# Fragment', '# Atom in Fragment', 'Mols in Fragment', 'XYZ', 'SMILES', 'Molecular Formulas']
        self.df = pd.DataFrame(columns = column_names)

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
        self.df['Time step [fs]'] = timesteps
        self.df['# Atom in Fragment'] = str_fragments
        self.df['# Fragment'] = frag_indices
        self.df['Mols in Fragment'] = self.elem_fragments
        self.df['XYZ'] = xyz_divided
        self.df['SMILES'] = self.smiles
        self.df['Molecular Formulas'] = self.mol_formulas

        index = 0
        for i in range(len(self.smiles)):
            for j in range(len(self.smiles[i])):
                if self.smiles[i][j] == ["Revise structure"]:
                    print("Time step: %.1i"%i)
                    print("Fragment index: %1i"%j)
                    index += 1
        print("\n--------Finished molecular transformations--------")
        print("There are %.1i molecules in total to be revised.\n" % index)

        self.df.to_csv("dataframe_nanosim.csv", sep = ';')

        return