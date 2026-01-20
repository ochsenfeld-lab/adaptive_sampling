import pickle
import seaborn as sns
import networkx as nx
from pyvis.network import Network

from .read_write_utils import read_reaction_list

class NanoNetwork(nx.DiGraph):
    """Class for Nanoreactor Networks

    This class defines networks obtained from ab initio reactor simulations and is a child class of nx.DiGraph, the NetworkX class for directd graphs.
    It extends the nx.DiGraph class by a lists for node and edge colors and mapping of SMILES on the node labels.

    Args:
        -
    """

    def __init__(self, **attr):
        super().__init__(**attr)
        self.graph = nx.DiGraph()
        self.node_labels = []
        self.node_colors = []
        self.edge_colors = []
        self.node_indices= {}

    def create_network(self, reactions_list: list=[]):
        nodes_list = []
        nodes_occurences = {}
        for i_elem, elem in enumerate(reactions_list):
            for i in range(2,4):
                for node_elem in elem[i]:
                    if node_elem not in nodes_list:
                        nodes_list.append(node_elem)
                        nodes_occurences[node_elem] = 1
                    else:
                        nodes_occurences[node_elem] += 1
                        
        edge_react = [] # list containing number of reactants so that we still know how many after construction of hypergraph
        hyperedges_list = []
        edges_list = []
        edges_labels = []

        time_edge_dict= {}
        reaction_edge_labels = []

        for index, elem in enumerate(reactions_list):
            edge_list=[]
            edge_react.append(len(elem[2]))

            reaction_edge_label = ''
            reaction_edge_label += ".".join(elem[2])
            reaction_edge_label += ">>"
            reaction_edge_label += ".".join(elem[3])
            
            hyperedge_tuple = (tuple(elem[2]), tuple(elem[3]))
            
            for react in elem[2]: # reactant index
                edge_list.append(react)
            for prod in elem[3]: # product index
                edge_list.append(prod)
                
            edge_tuple = tuple(edge_list)
            edges_labels.append(reaction_edge_label)
            edges_list.append(edge_tuple)
            hyperedges_list.append(hyperedge_tuple)
            reaction_edge_labels.append(reaction_edge_label)

            if edge_tuple not in time_edge_dict:
                time_edge_dict[edge_tuple] = [ts*0.5*50 for ts in elem[1]] 
            else:
                pass
                #time_edge_dict[edge_tuple].append(ts*0.5*50 for ts in elem[1])

        # define color palette based on event number
        RGB_tuples = sns.color_palette("Spectral", n_colors=len(edges_list))
        colors_hex = RGB_tuples.as_hex() 

        hypergraph = nx.DiGraph()
        hypergraph.add_edges_from(hyperedges_list)
        #self.graph.add_edges_from(hyperedges_list)

        colors_dict = {}
        for i_edge, edge in enumerate(hypergraph.edges):
            #print(i_edge)
            for node in edge:
                if node not in colors_dict:
                    colors_dict[node] = colors_hex[i_edge]
                hypergraph.add_node(node, color=colors_dict[node])

        # get some info about the graph
        print('''There are 
                \n%3i main nodes (representing unique molecular species) 
                \n%3i subgraphs (representing reaction paths) 
                \nin the original hypergraph.''' % (len(nodes_list), len(edges_list)))

        # make star-expansion on the hypergraph
        colors_dict = {}
        attrs = {}

        for i,node in enumerate(nodes_list):
            self.graph.add_node(node, size=25)
            self.node_labels.append(node)
            if node not in self.node_indices:
                self.node_indices[node] = i
            
        for i_edge, edge in enumerate(edges_list):
            for index, node in enumerate(edge):
                if node not in colors_dict:
                    colors_dict[node] = colors_hex[i_edge]

                reaction_label = reaction_edge_labels[i_edge]
                self.graph.add_node(reaction_label, color=colors_dict[node], size=10, label='%.1f ps'%(time_edge_dict[edge][0]/1000))
                
                if reaction_label not in colors_dict:
                    colors_dict[reaction_label] = colors_hex[i_edge]

                if index < edge_react[i_edge]:
                    self.graph.add_edge(node, reaction_label, weight=3, color=colors_dict[reaction_label])
                    if (node, reaction_label) not in attrs:
                        attrs[(node, reaction_label)] = {"time": time_edge_dict[edge]}
                else:
                    self.graph.add_edge(reaction_label, node, weight=3, color=colors_dict[reaction_label])
                    if (reaction_label, node) not in attrs:
                        attrs[(reaction_label, node)] = {"time": time_edge_dict[edge]}
                        #print(time_edge_dict[edge])
            
        self.node_colors = [colors_dict[node] for node in self.graph.nodes()]
        self.edge_colors = [self.graph[u][v]["color"] for u, v in self.graph.edges()]
        
        nx.set_edge_attributes(self.graph, attrs)
        nx.set_node_attributes(self.graph, colors_dict, name='color')
        #print(self.graph.nodes)
        #print(self.graph.edges)

        # Set font size for each node
        for node in self.graph.nodes():
            self.graph.nodes[node]['font'] = {'size': 20}

        # Set edge thickness by adding a 'width' attribute
        for u, v, data in self.graph.edges(data=True):
            data['width'] = data.get('weight', 1)  # use 'weight' or set custom width

        x = self.graph.nodes()
        print(type(x))
        pickle.dump(self.graph, open('star_exp_graph.pickle', 'wb'))
        pickle.dump(hypergraph, open('hypergraph.pickle', 'wb'))
                       
        print('''There are 
                \n%3i total nodes (including intermediate nodes containing reaction SMILES) 
                \n%3i edges 
                \nin the star-expansion graph.''' % (len(self.graph.nodes), len(self.graph.edges)))

        nt = Network(width='1000px', height='1000px', directed=True, notebook=True, cdn_resources='remote')#, bgcolor="#222222", font_color='#ffffff')#select_menu=True)#, bgcolor="#222222", font_color='#ffffff')
        nt.from_nx(self.graph)
        nt.show('graph.html', notebook=True)
        #nt.save_graph('graph.html')

        return
    
    def create_refined_network(self, reactions_list: list=[]):
        nodes_list = []
        nodes_occurences = {}
        for i_elem, elem in enumerate(reactions_list):
            for i in range(2,4):
                for node_elem in elem[i]:
                    if node_elem not in nodes_list:
                        nodes_list.append(node_elem)
                        nodes_occurences[node_elem] = 1
                    else:
                        nodes_occurences[node_elem] += 1
                        
        edge_react = [] # list containing number of reactants so that we still know how many after construction of hypergraph
        hyperedges_list = []
        edges_list = []
        edges_labels = []

        time_edge_dict= {}
        reaction_edge_labels = []
        transition_states = []
        reaction_energies = []

        for index, elem in enumerate(reactions_list):
            edge_list=[]
            edge_react.append(len(elem[2]))

            reaction_edge_label = ''
            reaction_edge_label += ".".join(elem[2])
            reaction_edge_label += ">>"
            reaction_edge_label += ".".join(elem[3])
            transition_states.append(elem[4]['ts'])
            reaction_energies.append({"rxn_energy": elem[4]['rxn_free_energy'], "rxn_barrier": elem[4]['rxn_barrier']})
            
            hyperedge_tuple = (tuple(elem[2]), tuple(elem[3]))
            
            for react in elem[2]: # reactant index
                edge_list.append(react)
            for prod in elem[3]: # product index
                edge_list.append(prod)
                
            edge_tuple = tuple(edge_list)
            edges_labels.append(reaction_edge_label)
            edges_list.append(edge_tuple)
            hyperedges_list.append(hyperedge_tuple)
            reaction_edge_labels.append(reaction_edge_label)

            if edge_tuple not in time_edge_dict:
                time_edge_dict[edge_tuple] = [ts*0.5*50 for ts in elem[1]] 
            else:
                pass
                #time_edge_dict[edge_tuple].append(ts*0.5*50 for ts in elem[1])

        # define color palette based on event number
        RGB_tuples = sns.color_palette("Spectral", n_colors=len(edges_list))
        colors_hex = RGB_tuples.as_hex() 

        hypergraph = nx.DiGraph()
        hypergraph.add_edges_from(hyperedges_list)
        #self.graph.add_edges_from(hyperedges_list)

        colors_dict = {}
        for i_edge, edge in enumerate(hypergraph.edges):
            #print(i_edge)
            for node in edge:
                if node not in colors_dict:
                    colors_dict[node] = colors_hex[i_edge]
                hypergraph.add_node(node, color=colors_dict[node])

        # get some info about the graph
        print('''There are 
                \n%3i main nodes (representing unique molecular species) 
                \n%3i subgraphs (representing reaction paths) 
                \nin the original hypergraph.''' % (len(nodes_list), len(edges_list)))

        # make star-expansion on the hypergraph
        colors_dict = {}
        attrs = {}

        for i,node in enumerate(nodes_list):
            self.graph.add_node(node, size=15)
            self.node_labels.append(node)
            if node not in self.node_indices:
                self.node_indices[node] = i
            
        for i_edge, edge in enumerate(edges_list):
            for index, node in enumerate(edge):
                if node not in colors_dict:
                    colors_dict[node] = colors_hex[i_edge]

                reaction_label = reaction_edge_labels[i_edge]
                #self.graph.add_node(reaction_label, color=colors_dict[node], size=5, label='%.1f ps'%(time_edge_dict[edge][1]/1000))
                self.graph.add_node(reaction_label, color=colors_dict[node], size=5, label=str(reaction_energies[i_edge]))

                if reaction_label not in colors_dict:
                    colors_dict[reaction_label] = colors_hex[i_edge]

                if index < edge_react[i_edge]:
                    self.graph.add_edge(node, reaction_label, color=colors_dict[reaction_label])
                    if (node, reaction_label) not in attrs:
                        attrs[(node, reaction_label)] = {"time": '%.1f ps'%(time_edge_dict[edge][1]/1000)}
                else:
                    self.graph.add_edge(reaction_label, node, color=colors_dict[reaction_label])
                    if (reaction_label, node) not in attrs:
                        attrs[(reaction_label, node)] = {"time": '%.1f ps'%(time_edge_dict[edge][1]/1000)}
            
        self.node_colors = [colors_dict[node] for node in self.graph.nodes()]
        self.edge_colors = [self.graph[u][v]["color"] for u, v in self.graph.edges()]
        
        #nx.set_edge_attributes(self.graph, attrs)
        nx.set_node_attributes(self.graph, colors_dict, name='color')
        print(self.graph.nodes)
        print(self.graph.edges)

        pickle.dump(self.graph, open('star_exp_graph.pickle', 'wb'))
        pickle.dump(hypergraph, open('hypergraph.pickle', 'wb'))
                       
        print('''There are 
                \n%3i total nodes (including intermediate nodes containing reaction SMILES) 
                \n%3i edges 
                \nin the star-expansion graph.''' % (len(self.graph.nodes), len(self.graph.edges)))

        nt = Network(width='1000px', height='1000px', directed=True, notebook=False, cdn_resources='remote')#, bgcolor="#222222", font_color='#ffffff')#select_menu=True)#, bgcolor="#222222", font_color='#ffffff')
        nt.from_nx(self.graph)
        nt.save_graph('graph.html')

        return
    
    def get_refined_network(self, ref_rxn_list: list=[]):
        # process refined reactions list [[# event, [ts1, ts2], [rct1, rct2, ...], [prod1, prod2, ...], dict({"rxn_free_energy": ..., "rxn_barrier": ..., "ts": [SMILES], "cat": {SMILES: ...})], [...], ..., [...]]

        hypergraph = nx.DiGraph()
        star_exp_graph = nx.DiGraph()

        #barriers = []

        for _, event in enumerate(ref_rxn_list):
            hypergraph.add_edge(tuple(event[2]), tuple(event[3]))
            reaction_label = 'rxn_barrier: %.1f kcal/mol'%(event[4]['rxn_barrier'])

            star_exp_graph.add_node(reaction_label, size=5, shape='*', color = float(event[4]['rxn_barrier']))
            for reactant in event[2]:
                star_exp_graph.add_node(reactant, label=reactant)
                star_exp_graph.add_edge(reactant, reaction_label, color = float(event[1][0]*50*0.5/1000))
                

            for product in event[3]:
                star_exp_graph.add_node(product, label=product)
                star_exp_graph.add_edge(reaction_label, product, color = float(event[1][1]*50*0.5/1000))

        labels = list(nx.get_node_attributes(star_exp_graph, "label").values())
        self.node_colors = list(nx.get_node_attributes(star_exp_graph, "color").values())
        self.edge_colors = [star_exp_graph[u][v]["color"] for u, v in star_exp_graph.edges()]
        #node_color_list = [star_exp_graph[]]

        #self.edge_colors = color_list   
        self.graph = star_exp_graph
        pickle.dump(self.graph, open('new_star_exp_graph.pickle', 'wb'))

        nt = Network(width='1000px', height='1000px', directed=True, notebook=False, cdn_resources='remote', select_menu=True)#, bgcolor="#222222", font_color='#ffffff')#select_menu=True)#, bgcolor="#222222", font_color='#ffffff')
        nt.from_nx(self.graph)
        nt.save_graph('new_star_exp_graph.html')

        return self.graph
    
    def get_mini_refined_network(self, ref_rxn_path: str=None, min_barrier: float=0.0, max_barrier: float=50.0, rneqp:bool = True):
        # process refined reactions list [[# event, [ts1, ts2], [rct1, rct2, ...], [prod1, prod2, ...], dict({"rxn_free_energy": ..., "rxn_barrier": ..., "ts": [SMILES], "cat": {SMILES: ...})], [...], ..., [...]]


        #barriers = []
        if ref_rxn_path == None:
            return "Please provide a path to the refined reactions list!"
        else:
            ref_rxn_list = read_reaction_list(ref_rxn_path)
            sel_ref_rxn_list = [event for event in ref_rxn_list if event[4]['rxn_barrier'] >= min_barrier and event[4]['rxn_barrier'] <= max_barrier and event[4]['energies'][1] <= event[4]['energies'][-1]]
            for elem in sel_ref_rxn_list:
                cat_dict = elem[4]['cat']
                cats = []
                for cat in cat_dict:
                    for cat_amount in range(cat_dict[cat]):
                        cats.append(cat)
                cats.sort()
                elem[4]['cat'] = cats    
                        
                elem[2].sort()
                elem[3].sort()
                #del elem[4]['cat']
            
            mini_indices={}
            for _, event in enumerate(sel_ref_rxn_list):
                catalyst = event[4]['cat']
                reaction_edge_label = ".".join(event[2]) + '>' + '.'.join(catalyst) + '>' + ".".join(event[3])
                if rneqp and event[2] == event[3]:
                    pass
                else:
                    if  reaction_edge_label not in mini_indices:
                        mini_indices[reaction_edge_label] = [event]
                    else:
                        for _, event_dict in enumerate(mini_indices[reaction_edge_label]):
                            if float(event_dict[4]['rxn_barrier']) >= float(event[4]['rxn_barrier']) and event_dict[4]['cat'] == catalyst:
                                event_dict[4]['rxn_barrier'] = float(event[4]['rxn_barrier'])
                                event_dict[4]['rxn_free_energy'] = float(event[4]['rxn_free_energy'])

            # do the graph part
            hypergraph = nx.DiGraph()
            star_exp_graph = nx.DiGraph()
            edge_coeffs = {}
            for _, key in enumerate(mini_indices):
                for event in mini_indices[key]:
                    hypergraph.add_edge(tuple(event[2]), tuple(event[3]))
                    reaction_label = 'rxn_barrier: %.6f kcal/mol'%(event[4]['rxn_barrier'])

                    star_exp_graph.add_node(reaction_label, size=5, shape='*', color = float(event[4]['rxn_barrier']))
                    for reactant in event[2]:
                        star_exp_graph.add_node(reactant, label=reactant)
                        star_exp_graph.add_edge(reactant, reaction_label, color = float(event[1][0]*50*0.5/1000))
                        if (reactant, reaction_label) not in edge_coeffs:
                            edge_coeffs[(reactant, reaction_label)] = 1
                        else:
                            edge_coeffs[(reactant, reaction_label)] += 1

                    for product in event[3]:
                        star_exp_graph.add_node(product, label=product)
                        star_exp_graph.add_edge(reaction_label, product, color = float(event[1][1]*50*0.5/1000))
                        if (reaction_label, product) not in edge_coeffs:
                            edge_coeffs[(reaction_label, product)] = 1
                        else:
                            edge_coeffs[(reaction_label, product)] += 1

                    for cat in event[4]['cat']:
                        star_exp_graph.add_node(cat, label=cat)
                        star_exp_graph.add_edge(cat, reaction_label, color = float(event[1][0]*50*0.5/1000))
                        star_exp_graph.add_edge(reaction_label, cat, color = float(event[1][0]*50*0.5/1000))
                        if (cat, reaction_label) not in edge_coeffs:
                            edge_coeffs[(cat, reaction_label)] = 1
                        else:
                            edge_coeffs[(cat, reaction_label)] += 1

                        if (reaction_label, cat) not in edge_coeffs:
                            edge_coeffs[(reaction_label, cat)] = 1
                        else:
                            edge_coeffs[(reaction_label, cat)] += 1



            labels = list(nx.get_node_attributes(star_exp_graph, "label").values())
            self.node_colors = list(nx.get_node_attributes(star_exp_graph, "color").values())
            self.edge_colors = [star_exp_graph[u][v]["color"] for u, v in star_exp_graph.edges()]
            #edge_coeffs = list(nx.get_edge_attributes(star_exp_graph, "weight").values())
            #node_color_list = [star_exp_graph[]]

            #self.edge_colors = color_list   
            self.graph = star_exp_graph
            pickle.dump(self.graph, open('new_star_exp_graph.pickle', 'wb'))

            nt = Network(width='1000px', height='1000px', directed=True, notebook=False, cdn_resources='remote', select_menu=True)#, bgcolor="#222222", font_color='#ffffff')#select_menu=True)#, bgcolor="#222222", font_color='#ffffff')
            nt.from_nx(self.graph)
            nt.save_graph('new_star_exp_graph.html')

            return self.graph, edge_coeffs
