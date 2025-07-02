import os
import json
from ast import literal_eval
from .read_write_utils import get_reaction_traj, read_reaction_list
import networkx as nx
import numpy as np
from itertools import product

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in sorted(filenames):
            yield os.path.abspath(os.path.join(dirpath, f))

def get_reaction_patterns(pattern_json: str, freq_threshold: int = 25):
    
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

def search_pattern(pattern: str, patterns_list: list):
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

def check_if_whole_pattern(pattern: list, reaction_collection: list) -> bool:
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
    #print(reactions_pattern)

    #reactions_pattern = [[reaction[0], reaction[1]] for i_reaction, reaction in enumerate(pattern)]

    if all(element in reaction_collection for element in reactions_pattern):
        check_is_subset = True
    else:
        check_is_subset = False
    
    return check_is_subset

def extract_reactions(root_dir: str,  patterns: list, rct_indices: list):
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
                #print(f"Pattern: {i_pattern}")
                pattern = patterns[i_pattern]
                event_pattern_list = []
                atomic_indices = []
                if check_if_whole_pattern(pattern, reaction_collection):
                    #print("yay")
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
                        #for event_elem in event_indices:
                        event_pattern_list.append(event_indices)

                    event_collection = list(product(*event_pattern_list))

                    # Convert tuples to lists if needed
                    event_collection = [list(tup) for tup in event_collection]

                    for event_list in event_collection:
                        #print("Events: " + str(event_list))
                        atomic_indices = []
                        for event in event_list:
                            #print(reaction_list[event - 1])
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
                                    f"{root_dir}/extracted_missing/pattern{i_pattern}_traj_sim{index_sim}")

                                else: # this is the case where (found is True, but isomerization)
                                    for _, event in enumerate(event_group):
                                        get_reaction_traj(traj_file, df_file, reaction_file, [event], 
                                        f"{root_dir}/extracted_missing/pattern{i_pattern}_traj_sim{index_sim}")
                            else:
                                get_reaction_traj(traj_file, df_file, reaction_file, event_group, 
                                f"{root_dir}/extracted_missing/pattern{i_pattern}_traj_sim{index_sim}")
                                
    print("DONE!")

def find_conn_events(event_list, atomic_indices):
    # Create a graph to find events sharing atoms
    G = nx.Graph()

    # Add nodes
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

def extract_first_last_structure(input_file, first_output, last_output):

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
