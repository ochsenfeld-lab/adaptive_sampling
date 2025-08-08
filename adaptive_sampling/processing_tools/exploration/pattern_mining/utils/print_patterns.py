import sys, os
import json 
import networkx as nx
from collections import defaultdict


with open(sys.argv[1], 'r') as file:
    data = json.load(file)

for pattern in data:
    if len(pattern) > 2: print("Pattern with frequencies", pattern[1], pattern[2], "(significant)" if pattern[3] else "(not significant)")
    else: print("Pattern with frequency", pattern[1])
    hypergraph = pattern[0] # hypergraph pattern as list of hyperedges
    G = nx.Graph() # star expansion
    for hyperedge in hypergraph:
        tail = hyperedge[0] # list of reagents 
        head = hyperedge[1] # list of products 
        coef_tail = hyperedge[2] # coefficients of reagents
        coef_head = hyperedge[3] # coefficients of products 
        print(tail, head, coef_tail, coef_head)
        # star expansion of hyperedge
        hyperedge_key = str(hyperedge)
        G.add_node( hyperedge_key ) 
        for specie in tail:
            G.add_node(specie)
            G.add_edge(hyperedge_key, specie)
        for specie in head:
            G.add_node(specie)
            G.add_edge(hyperedge_key, specie)
        
    print("Star expansion: ", G)
    print("")
