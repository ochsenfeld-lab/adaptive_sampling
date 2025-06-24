import pandas as pd

import seaborn as sns
import networkx as nx

from .utils import *
from .read_write_utils import *
from .nanosim_class import *

# function to find fragment index for atom and ts
def find_reactions(df: pd.DataFrame, atom_r: str, ts: int, step:int = 80) -> Tuple[list, list, list]:
    ''' Find reaction by comparing fragments containing given atom indexa t time step ts and ts + step.

    Args:
        df: data frame containing all necessary information from nanoreactor simulation
        atom_r: atom index of reactive atom to be searched for
        ts: reactant time step
        step: interval between reactants and products 

    Returns:
        atom_indices_reactant_list_sorted: list of sorted atom indices present in the reaction
        [atom_indices_reactant_list, atom_indices_product_list]: list containing the reaction expressed with atom indices
        [smiles_reactant_list, smiles_product_list]: list containing the reaction expressed in SMILES
    '''
    atom_indices_reactant_list = []
    atom_indices_product_list = []
    smiles_reactant_list = []
    smiles_product_list = []
    
    # find product
    for fragment in range(len(df['# Fragment'][ts+step])):
        if atom_r in df['# Atom in Fragment'][ts+step][fragment]:
            atom_indices_product_list.append(df['# Atom in Fragment'][ts+step][fragment])
            for elem in df['SMILES'][ts+step][fragment]:
                smiles_product_list.append(elem)
            break
    
    # look up atoms in the product and add fragments
    for atom_index in atom_indices_product_list[0]:
        for fragment in range(len(df['# Fragment'][ts])):
            if atom_index in df['# Atom in Fragment'][ts][fragment]:
                if df['# Atom in Fragment'][ts][fragment] not in atom_indices_reactant_list:
                    atom_indices_reactant_list.append(df['# Atom in Fragment'][ts][fragment])
                    for elem in df['SMILES'][ts][fragment]:
                        smiles_reactant_list.append(elem)
    
    # return early if no reaction found
    if smiles_reactant_list == smiles_product_list:
        return 
    
    # transform atom_indices_lists into merged lists of sorted integers
    atom_indices_reactant_list_sorted = sort_merge_str_list_to_int_list(atom_indices_reactant_list)
    atom_indices_product_list_sorted = sort_merge_str_list_to_int_list(atom_indices_product_list)
    
    # look for missed atoms in the reactant and product lists in a loop
    # maybe use set instead of list to avoid sort
    while atom_indices_reactant_list_sorted != atom_indices_product_list_sorted:
        for reactant_index in range(len(atom_indices_reactant_list)):
            for atom_index in atom_indices_reactant_list[reactant_index]:
                for fragment in range(len(df['# Fragment'][ts+step])):
                    if atom_index in df['# Atom in Fragment'][ts+step][fragment]:
                        if df['# Atom in Fragment'][ts+step][fragment] not in atom_indices_product_list:
                            atom_indices_product_list.append(df['# Atom in Fragment'][ts+step][fragment])
                            for elem in df['SMILES'][ts+step][fragment]:
                                smiles_product_list.append(elem)
                                
                            atom_indices_product_list_sorted = sort_merge_str_list_to_int_list(atom_indices_product_list)
                            if atom_indices_reactant_list_sorted == atom_indices_product_list_sorted:
                                break
                                
            for product_index in range(len(atom_indices_product_list)):
                for atom_index in atom_indices_product_list[product_index]:
                    for fragment in range(len(df['# Fragment'][ts])):
                        if atom_index in df['# Atom in Fragment'][ts][fragment]:
                            if df['# Atom in Fragment'][ts][fragment] not in atom_indices_reactant_list:
                                atom_indices_reactant_list.append(df['# Atom in Fragment'][ts][fragment])
                                for elem in df['SMILES'][ts][fragment]:
                                    smiles_reactant_list.append(elem)
                                
                                atom_indices_reactant_list_sorted = sort_merge_str_list_to_int_list(atom_indices_reactant_list)
                                if atom_indices_reactant_list_sorted == atom_indices_product_list_sorted:
                                    break
    
    # exclude atom switches
    if smiles_reactant_list == smiles_product_list:
        return    
    
    return atom_indices_reactant_list_sorted, [atom_indices_reactant_list, atom_indices_product_list], [smiles_reactant_list, smiles_product_list]

def construct_reactions_list(df: pd.DataFrame, start_ts_index: int = 19, period_ts_steps: int = 80) -> list:
    ''' Get list of reactions to be able to construct network.

    Args:
        df: data frame containing all necessary information from nanoreactor simulation
        start_ts_index: specify index of time step at which the first search should be conducted
                        Per default this is set for the smooth step spherical constraint function at the end of the expansion period.
        period_ts_steps: step width to search for products, corresponds to period of the confinement function
    Returns:
        reactions_list: list of reactions\n
                Format: [event #, [ts_r, ts_p], [smiles_r...], [smiles_p...]]\n
    '''

    # ts # = 19 --> always look at steps at the end of the expansion (sin_cos: 19*50*0.5=475, at 500 fs the contraction starts)
    events = []
    time_steps = []
    reactions = []
    reactions_list = []
    atom_indices = []
    
    natoms = sum([len(listElem) for listElem in df['# Atom in Fragment'][0]])
    print(natoms)

    event_counter = 0
   
    # first event

    indices_ts=[]
    reactions_ts=[]
    reactants_ts=[]
    products_ts=[]

    atom_list_sorted = []
    print("Time Step: " + str(0) + " -> " + str(start_ts_index))
    for atom_index in range(natoms):
        if atom_index+1 not in atom_list_sorted:
            try:
                atom_list_sorted,atom_index_list,smiles_reaction = find_reactions(df, str(atom_index + 1), 0, start_ts_index)
                set_reactants = set(smiles_reaction[0])
                set_products = set(smiles_reaction[1])
                if [set_reactants,set_products] not in reactions:
                    event_counter += 1
                    events.append(event_counter)
                    time_steps.append([0, start_ts_index])
                    reactions.append([set_reactants,set_products])
                    atom_indices.append(atom_index_list)

                    reactions_list.append(["Event " + str(event_counter), [0, start_ts_index], smiles_reaction[0], smiles_reaction[1], atom_list_sorted])
                    print("# Event: " + str(event_counter))
                    print(str(smiles_reaction[0]) + " -> " + str(smiles_reaction[1]))

            except:
                pass

    for ts in range(start_ts_index, len(df['Time step [fs]'])-period_ts_steps, period_ts_steps):
        indices_ts=[]
        reactions_ts=[]
        reactants_ts=[]
        products_ts=[]

        atom_list_sorted = []
        print("Time Step: " + str(ts) + " -> " + str(ts + period_ts_steps))
        for atom_index in range(natoms):
            if atom_index+1 not in atom_list_sorted:
                try:
                    atom_list_sorted,atom_index_list,smiles_reaction = find_reactions(df, str(atom_index + 1), ts, period_ts_steps)
                    set_reactants = set(smiles_reaction[0])
                    set_products = set(smiles_reaction[1])
                    if [set_reactants,set_products] not in reactions:
                        event_counter += 1
                        events.append(event_counter)
                        time_steps.append([ts,ts + period_ts_steps])
                        reactions.append([set_reactants,set_products])
                        atom_indices.append(atom_index_list)

                        reactions_list.append(["Event " + str(event_counter), [ts, ts + period_ts_steps], smiles_reaction[0], smiles_reaction[1], atom_list_sorted])
                        print("# Event: " + str(event_counter))
                        print(str(smiles_reaction[0]) + " -> " + str(smiles_reaction[1]))

                except:
                    pass
    print("Number of events found: %3i" % (event_counter))

    with open("reactions_list.json", "w") as fp:
        json.dump(reactions_list, fp)

    return reactions_list

class NanoNetwork(nx.DiGraph):
    ''' Class for Nanoreactor Networks

    This class defines networks obtained from ab initio nanoreactor simulations and is a child class of nx.DiGraph, the NetworkX class for directd graphs.
    It extends the nx.DiGraph class by a lists for node and edge colors and mapping of SMILES on the node labels.

    Args:
        -
    '''
    def __init__(self, **attr):
        super().__init__(**attr)
        self.graph = nx.DiGraph()
        self.node_labels = []
        self.node_colors = []
        self.edge_colors = []

    def create_network(self, reactions_list: list):
        ''' Construct network from already computed reaction list.

        Args:
            reactions_list: list of reactions\n
                    Format: [event #, [ts_r, ts_p], [smiles_r...], [smiles_p...]]\n
        Returns:
            G: simulation graph (consecutive monomolecular transformations have been excluded)
        '''

        # Calculate how many distinct time steps there are
        time = 0
        length = 0
        for elem in reactions_list:
            if elem[1][0] >= time:
                length += 1
                time = elem[1][0]

        RGB_tuples = sns.color_palette("Spectral", n_colors=length)
        hex_codes = RGB_tuples.as_hex()

        index=0
        time_color_dict={}
        for elem in reactions_list:
            if elem[1][0] not in time_color_dict:
                time_color_dict[elem[1][0]] = hex_codes[index]
                index+=1

        node_color_dict = {}
        node_label_dict = {}
        nodes_list = []
        time_list = []
        node_label = 0
        
        # Make nodes list
        for i in range(len(reactions_list)):
            elem_length = len(reactions_list[i])
            for elem_index in range(2,elem_length - 1):
                for elem in reactions_list[i][elem_index]:
                    if elem not in nodes_list:
                        nodes_list.append(elem)
                    if elem not in node_color_dict:
                        node_color_dict[elem] = time_color_dict[reactions_list[i][1][0]]
                        self.node_labels.append(elem)
                        node_label_dict[elem] = node_label
                        node_label += 1
        
        for i,elem in enumerate(nodes_list):
            self.graph.add_node(elem, color = node_color_dict[elem])

        # create edges
        for elem in reactions_list:
            elem_length = len(elem)
            for i in range(len(elem[2])): # reactant index
                for j in range(len(elem[3])): # product index
                    if elem[2][i] != elem[3][j]:
                        self.graph.add_edge(elem[2][i], elem[3][j], color=node_color_dict[elem[2][i]])


        n_seen=[]
        n_to_remove = []
        edges_q = list(self.graph.edges())
        while len(edges_q) != 0:
            edges_popped = edges_q.pop(0)
            n = edges_popped[0]
            if n in n_seen:
                continue
            else:
                n_seen.append(n)
            if len(self.graph.in_edges(n)) == 1 and len(self.graph.out_edges(n)) == 1:
                up_node = [elem for elem in self.graph.predecessors(n)][0]
                down_node = [elem for elem in self.graph.successors(n)][0]
                if up_node != down_node:
                    self.graph.add_edge(up_node,down_node,color=node_color_dict[n])
                    self.graph.remove_edge(up_node,n)
                    self.graph.remove_edge(n,down_node)
                    n_to_remove.append(n)
            else:
                continue

        self.graph.remove_nodes_from(n_to_remove)
    
        for node in self.graph.nodes():
            self.node_colors.append(node_color_dict[node])
        
        self.graph = nx.relabel_nodes(self.graph, node_label_dict)
        
        edges = self.graph.edges()
        self.edge_colors = [self.graph[u][v]['color'] for u,v in edges]

        return