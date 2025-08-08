import sys, os
import json


def is_subset(smaller, larger):
    it = iter(larger)
    return all(elem in it for elem in smaller)


with open(sys.argv[1], 'r') as file:
    data = json.load(file)

hyperedge2id = {}


itemset2patterns = {}
for pattern in data:
    hypergraph = pattern[0] # hypergraph pattern as list of hyperedges
    itemset = []
    for edge in hypergraph:
        e = str(edge[0])+'~'+str(edge[1])+'~'+str(edge[2])+'~'+str(edge[3])
        if e not in hyperedge2id: hyperedge2id[e] = len(hyperedge2id)
        itemset.append( hyperedge2id[e] )
    itemset = sorted(itemset)
    itemset2patterns[str(itemset)] = pattern

children = {}
count_parents = {}

for itemset in itemset2patterns.keys():
    for itemset2 in itemset2patterns.keys():
        if itemset == itemset2: continue
        if itemset not in count_parents: count_parents[itemset] = 0
        if itemset2 not in children: children[itemset2] = []
        
        if is_subset(eval(itemset), eval(itemset2)):
            children[itemset2].append(itemset)
            count_parents[itemset] +=1


for itemset in itemset2patterns.keys():
    if count_parents[itemset] == 0:
        pattern = itemset2patterns[itemset]
        print(pattern)
        print("Sub-patterns")
        for sub in children[itemset]:
            subpattern = itemset2patterns[sub]
            print(subpattern)
        print()
