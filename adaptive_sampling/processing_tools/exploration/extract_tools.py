import json
from ast import literal_eval
from .read_write_utils import get_reaction_traj, read_reaction_list
import networkx as nx
import numpy as np
from itertools import product
from .utils import absoluteFilePaths
import os
from pathlib import Path
import shutil
from typing import Tuple

def get_reaction_patterns(
    pattern_json: str,
    freq_threshold: int = 25
) -> Tuple[list, dict, list, dict, list]:
    """Helper function to read reaction patterns from a JSON file and filter them based on frequency.

    Args:
        pattern_json: Path to the JSON file containing reaction patterns and their frequencies.
        freq_threshold: Minimum frequency for a pattern to be included in the output.

    Returns:
        -
    """
    with open(pattern_json, 'r') as file:
        data = json.load(file)
    
    complete_rxn_set = []
    small_rxn_set = []
    small_rxn_dict = {}
    rxn_freq_dict = {}
    pattern_indices = []
    
    for i,pattern in enumerate(data):
        freq = pattern[1] # frequency
        small_hypergraph = pattern[0]
        rxns = []
        for hyperedge in small_hypergraph:
            reactants = []
            for i_react, coeff in enumerate(hyperedge[2]):
                for coeff_reactant in range(coeff):
                    reactants.append(hyperedge[0][i_react])
            products = []
            for i_prod, coeff in enumerate(hyperedge[3]):
                for coeff_product in range(coeff):
                    products.append(hyperedge[1][i_prod])
                
            rxns.append([reactants, products])        
        
        complete_rxn_set.append(small_hypergraph)
        if freq >= freq_threshold:
            pattern_indices.append(i)
            small_hypergraph.sort()
            small_rxn_dict[str(small_hypergraph)] = i
            small_rxn_set.append(small_hypergraph)
            print("Frequency of pattern #%2i: %2i" % (i,freq))
            print("Pattern: %s" % (small_hypergraph))
            print("%s \n" % (rxns))
        
            if freq not in rxn_freq_dict:
                rxn_freq_dict[freq] = [str(small_hypergraph)]
            else:
                rxn_freq_dict[freq].append(str(small_hypergraph))

    return small_rxn_set, small_rxn_dict, complete_rxn_set, rxn_freq_dict, pattern_indices

def search_pattern(
    pattern: str, 
    patterns_list: list
) -> Tuple[int, bool]:
    """Search for a specific reaction pattern in a list of patterns.
    
    Args:
        pattern: The reaction pattern to search for, represented as a string.
        patterns_list: A list of reaction patterns, where each pattern is a list of reactions.
    
    Returns:
        A tuple containing the index of the found pattern and a boolean indicating if it was found.
    """
    found = False
    index = -1
    rct = literal_eval(pattern)
    rct.sort()
    if patterns_list != None:
        for j, reaction in enumerate(patterns_list):
            reaction.sort()
            if rct == reaction:
                return j, True
            else:
                pass
    return index,found

def check_if_whole_pattern(
    pattern: list, 
    reaction_collection: list
)-> bool:
    """Check if all reactions in a given pattern are present in a collection of reactions.
    
    Args:
        pattern: A list of reactions, where each reaction is represented as a list containing reactants and products.
        reaction_collection: A list of reactions to check against, formatted similarly to the pattern.

    Returns:
        A boolean indicating whether all reactions in the pattern are found in the reaction collection.
    """
    check_is_subset = False
    reactions_pattern = []
    for i_reaction, reaction in enumerate(pattern):
        reactants = []
        for i_react, coeff in enumerate(reaction[2]):
            for coeff_reactant in range(coeff):
                reactants.append(reaction[0][i_react])
        products = []
        for i_prod, coeff in enumerate(reaction[3]):
            for coeff_product in range(coeff):
                products.append(reaction[1][i_prod])
        reactants.sort()
        products.sort()
        reactions_pattern.append([reactants,products])

    if all(element in reaction_collection for element in reactions_pattern):
        check_is_subset = True
    else:
        check_is_subset = False
    
    return check_is_subset

def extract_reactions(
    root_dir: str,
    patterns: list, 
    rct_indices: list
) -> None:
    """Extracts reaction trajectories based on specified reaction patterns from simulation data.
    
    Args:
        root_dir: The root directory containing subdirectories with the reaction lists, dataframes, and trajectories.
        patterns: A list of reaction patterns to search for.
        rct_indices: Indices of the reaction patterns to be extracted.
        
    Returns:
        None
    """
    if not os.path.isdir("extracted"):
        print(f"Creating directory: extracted")
        os.makedirs("extracted", exist_ok=True)
    else:
        print(f"Directory already exists: extracted")


    if not os.path.isdir("reactions_lists") or not os.path.isdir("dfs") or not os.path.isdir("trajs"):
        print("Please move the reactions lists to a directory named reactions_lists before starting and make sure that dfs/ and trajs/ are available in this folder!")
        return

    for reaction_file in absoluteFilePaths(f"{root_dir}/reactions_lists/"):
        reaction_list = read_reaction_list(reaction_file)   
        index_sim = reaction_file[-8:-5]
        reaction_collection = []
        if reaction_list != []:
            print(reaction_file[-23:])
            for i_event, event in enumerate(reaction_list):
                event[2].sort()
                event[3].sort()
                reaction_collection.append([event[2],event[3]]) # we have to transform evth into sets to
                                                                          # to make sure we find the patterns and 
                                                                          # match them
            for i_pattern in rct_indices:
                pattern = patterns[i_pattern]
                event_pattern_list = []
                atomic_indices = []
                if check_if_whole_pattern(pattern, reaction_collection):
                    for i_reaction, reaction in enumerate(pattern):
                        reactants = []
                        for i_react, coeff in enumerate(reaction[2]):
                            for coeff_reactant in range(coeff):
                                reactants.append(reaction[0][i_react])
                        products = []
                        for i_prod, coeff in enumerate(reaction[3]):
                            for coeff_product in range(coeff):
                                products.append(reaction[1][i_prod])
                        reactants.sort()
                        products.sort()
                        event_indices = [i + 1 for i, elem in enumerate(reaction_collection) if 
                                          [reactants, products] == elem]
                        event_pattern_list.append(event_indices)

                    event_collection = list(product(*event_pattern_list))

                    # Convert tuples to lists if needed
                    event_collection = [list(tup) for tup in event_collection]

                    for event_list in event_collection:
                        atomic_indices = []
                        for event in event_list:
                            atomic_indices.append(set(reaction_list[event - 1][4]))

                        event_groups, atomic_indices_groups = find_conn_events(event_list, atomic_indices)
                    
                        for i_group, event_group in enumerate(event_groups):
                            traj_file = f"{root_dir}/trajs/traj_{index_sim}.xyz"
                            df_file = f"{root_dir}/dfs/dataframe_nanosim_{index_sim}.csv"
                            
                            if len(event_group) > 1:
                                iso = True
                                found = True
                                #print(atomic_indices_groups[i_group])
                                common = set(atomic_indices_groups[i_group][0]).intersection(*atomic_indices_groups[i_group])
                                for elem in atomic_indices_groups[i_group]:
                                    if common != elem:
                                        iso = False
                                        break
                                if found and not iso:
                                    get_reaction_traj(traj_file, df_file, reaction_file, event_group, 
                                    f"{root_dir}/extracted/pattern{i_pattern}_traj_sim{index_sim}")

                                else: # this is the case where (found is True, but isomerization)
                                    for _, event in enumerate(event_group):
                                        get_reaction_traj(traj_file, df_file, reaction_file, [event], 
                                        f"{root_dir}/extracted/pattern{i_pattern}_traj_sim{index_sim}")
                            else:
                                get_reaction_traj(traj_file, df_file, reaction_file, event_group, 
                                f"{root_dir}/extracted/pattern{i_pattern}_traj_sim{index_sim}")

    os.chdir("extracted/")
    extracted_dir = Path(os.getcwd())
    for txt_file in extracted_dir.glob("*.txt"):
        base = txt_file.stem  # filename without extension
        xyz_file = extracted_dir / f"{base}.xyz"

        # create unique folder for this pair
        target_dir = extracted_dir / base
        target_dir.mkdir(exist_ok=True)

        # rename txt with prefix
        new_txt_name = f"charge_mult_{txt_file.name}"
        shutil.move(str(txt_file), target_dir / new_txt_name)

        # move xyz if it exists
        if xyz_file.exists():
            shutil.move(str(xyz_file), target_dir / xyz_file.name)

    print("DONE!")

def find_conn_events(
    event_list: list,
    atomic_indices: list
) -> Tuple[list, list]:
    """Find connected events based on shared atomic indices.

    Args:
        event_list: A list of events, where each event is represented by its index.
        atomic_indices: A list of sets, where each set contains atomic indices involved in the corresponding event.
    Returns:
        A tuple containing:
            - A list of groups of events that share atomic indices.
            - A list of groups of atomic indices corresponding to the event groups.
    """
    # Create a graph to find events sharing atoms
    G = nx.Graph()
    G.add_nodes_from(range(len(atomic_indices)))

    # Add edges between reactions that share atoms
    for i in range(len(atomic_indices)):
        for j in range(i + 1, len(atomic_indices)):
            if atomic_indices[i] & atomic_indices[j]:  # Check for intersection
                G.add_edge(i, j)

    # Find connected components (groups of reactions sharing atoms)
    groups = list(nx.connected_components(G))
    event_groups = [[event_list[i] for i in group] for group in groups]
    atomic_indices_groups = [[atomic_indices[i] for i in group] for group in groups]

    return event_groups, atomic_indices_groups

def extract_reactant_product(
    input_file: str,
    first_output: str="start.xyz", 
    last_output:str="end.xyz"
) -> None:
    """Extracts the first and last structures from a multi-structure XYZ file.

    Args:
        input_file: Path to the input XYZ file containing multiple structures.
        first_output: Name of the file to be created which contains the dirst structure.
        last_output: Name of the file to be created which contains the last structure.
        
    Returns:
        None
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    structures = []
    current_structure = []

    for line in lines:
        if line.strip().isdigit():  # New structure starts when a number appears
            if current_structure:
                structures.append(current_structure)
            current_structure = [line]
        else:
            current_structure.append(line)

    if current_structure:  # Append the last structure
        structures.append(current_structure)

    # Write the first structure
    with open(first_output, "w") as f:
        f.writelines(structures[0])

    # Write the last structure
    with open(last_output, "w") as f:
        f.writelines(structures[1])
