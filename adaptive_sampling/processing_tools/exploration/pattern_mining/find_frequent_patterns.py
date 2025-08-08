import sys, os
import json 
import networkx as nx
from collections import defaultdict



# Step 1: translate reactions into (item)sets of integers
species2id = {}
hyperedge2id = {}
with open(sys.argv[2]+'.trns', 'w') as out:
    out.write("")

cnt = 0

print(len(list(os.listdir(sys.argv[1]))))
for filename in os.listdir(sys.argv[1]):
    with open(os.path.join(sys.argv[1], filename)) as json_data:
        data = json.load(json_data)

    transaction = []
    for event in data:
        tail = str(sorted(list(set(event[2]))))
        head = str(sorted(list(set(event[3]))))
        coef_tail = str([ event[2].count(x) for x in sorted(list(set(event[2]))) ])
        coef_head = str([ event[3].count(x) for x in sorted(list(set(event[3]))) ])
        hyperedge = tail+'~'+head+'~'+coef_tail+'~'+coef_head

        if tail not in species2id:
            species2id[tail] = len(species2id)
        if head not in species2id:
            species2id[head] = len(species2id)
        if hyperedge not in hyperedge2id:
            hyperedge2id[hyperedge] = len(hyperedge2id)

        transaction.append(hyperedge2id[hyperedge])
    
    transaction.sort()
    with open(sys.argv[2]+'.trns', 'a') as out:
        for item in transaction:
            out.write(str(item)+" ")
        out.write(" \n")


# Step 2: use a frequent itemset miner to obtain patterns
# https://research.nii.ac.jp/~uno/code/lcm.html
frequency = int(sys.argv[3])
os.system(f"miners/lcm1 {sys.argv[2]}.trns {frequency}  > {sys.argv[2]}.frq")


# Step 3: translate back the patterns to reactions
id2hyperedge = {v: k for k, v in hyperedge2id.items()}
itemsets = []
frequencies = []

# get raw frequent hypergraph patterns 
with open(sys.argv[2]+'.frq', 'r') as file:
    for line in file:
        toks = line.rstrip().split('(')
        freq = int(toks[1][:-1])

        itemset = toks[0].strip().split(' ')
        itemset = sorted( [ int(itm) for itm in itemset if (len(itm)>0)] )
        if len(itemset) == 0: continue 

        itemsets.append(itemset)
        frequencies.append(freq)


# filter out disconnected ones
connected = []
frequencies_conn = []
graphs_conn = []

for i in range(len(itemsets)):
    itemset = itemsets[i] # this is one pattern
    itemset = [ id2hyperedge[itm] for itm in itemset ]
    G = nx.Graph()
    for item in itemset:
        G.add_node(item)
        tail, head, coef_tail, coef_head = item.split('~')
        tail = eval(tail)
        head = eval(head)
        for specie in tail:
            G.add_node(specie)
            G.add_edge(item, specie)
        for specie in head:
            G.add_node(specie)
            G.add_edge(item, specie)
    
    if nx.is_connected(G):
        connected.append(itemsets[i])
        graphs_conn.append(G)
        frequencies_conn.append(frequencies[i])
    

# filter to keep just closed itemsets
def is_subset(smaller, larger):
    it = iter(larger)
    return all(elem in it for elem in smaller)

freq_bins = defaultdict(list)
for i in range(len(connected)):
    freq_bins[frequencies_conn[i]].append(i)


closed_indexes = []  
freqs = list(freq_bins.keys())
freqs.sort(key=lambda x: -x)

for freq in freqs:
    ids = freq_bins[freq]
    # Sort itemsets by length (shortest first)
    ids.sort(key=lambda i: len(connected[i]))
    itemsets = [ connected[i] for i in ids]


    for i, itemset in enumerate(itemsets):
        is_closed = True
        for j in range(i + 1, len(itemsets)):
            if is_subset(itemset, itemsets[j]):
                is_closed = False
                break
        
        if is_closed:
            closed_indexes.append( ids[i] )
        else:
            pass

output = []
for idx in closed_indexes:
    reactions = [x for x in graphs_conn[idx].nodes if ('~' in x)]
    hyperedges_out = []
    for reaction in reactions:
        tail, head, coef_tail, coef_head = reaction.split('~')
        tail = eval(tail)
        head = eval(head)
        coef_tail = eval(coef_tail)
        coef_head = eval(coef_head)
        hyperedges_out.append( (tail, head, coef_tail, coef_head) )
    output.append( (hyperedges_out, frequencies_conn[idx]) )


# Step 4: output patterns, remove temp files
with open(sys.argv[2]+'.out.json', 'w') as out:
    json.dump(output, out)

os.system(f'rm {sys.argv[2]}.trns {sys.argv[2]}.frq')