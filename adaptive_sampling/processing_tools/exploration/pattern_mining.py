import ast
import os, shutil
import json 
import networkx as nx
from collections import defaultdict
import warnings
from math import log, lgamma
from scipy.stats import fisher_exact
from scipy.stats.contingency import odds_ratio

import statsmodels

def find_freq_patterns(
    rct_folder: str, 
    collection_name: str, 
    frequency: int
) -> None:
    """Find frequent hypergraph patterns in a collection of reactions.
    
    Args:
        rct_folder (str): Path to the folder containing reaction_list JSON files.
        collection_name (str): Base name for output files.
        frequency (int): Minimum frequency threshold for patterns to be considered frequent, i.e., number of distinct simulations they appear in.
        
    Returns:
        None. Outputs results to `<collection_name>.out.json`.
    """   
    miner_path = shutil.which("lcm1")
    if miner_path is None:
        warnings.warn(f"Required mining program lcm1 not found in PATH.", UserWarning)
        raise RuntimeError(
            f"lcm1 is required but not installed or not in PATH. "
            f"Please install and compile it before running this package."
        )

    # Step 1: translate reactions into (item)sets of integers
    species2id = {}
    hyperedge2id = {}
    with open(collection_name+'.trns', 'w') as out:
        out.write("")

    print(f"Number of simulations found: {len(list(os.listdir(rct_folder)))}")
    for filename in os.listdir(rct_folder):
        with open(os.path.join(rct_folder, filename)) as json_data:
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

            if hyperedge2id[hyperedge] not in transaction:
                transaction.append(hyperedge2id[hyperedge])
        
        transaction.sort()
        with open(collection_name+'.trns', 'a') as out:
            for item in transaction:
                out.write(str(item)+" ")
            out.write(" \n")


    # Step 2: use a frequent itemset miner to obtain patterns
    # https://research.nii.ac.jp/~uno/code/lcm.html
    frequency = int(frequency)
    os.system(f"lcm1 {collection_name}.trns {frequency}  > {collection_name}.frq")


    # Step 3: translate back the patterns to reactions
    id2hyperedge = {v: k for k, v in hyperedge2id.items()}
    itemsets = []
    frequencies = []

    # get raw frequent hypergraph patterns 
    with open(collection_name+'.frq', 'r') as file:
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
    with open(collection_name+'.out.json', 'w') as out:
        json.dump(output, out)

    os.system(f'rm {collection_name}.trns {collection_name}.frq')


def find_freq_patterns_pair(
    rct_folder1: str, 
    rct_folder2: str, 
    collection_name: str, 
    frequency: int
) -> None:
    """Find frequent hypergraph patterns in two collections of reactions so they share the same indices and can be compared more easily.
    Args:
        rct_folder1 (str): Path to the folder containing reaction_list JSON files for the first collection.
        rct_folder2 (str): Path to the folder containing reaction_list JSON files for the second collection.
        collection_name (str): Base name for output files.
        frequency (int): Minimum frequency threshold for patterns to be considered frequent, i.e., number of distinct simulations they appear in.
        
    Returns:
        None. Outputs results to `<collection_name>.out_1.json` and `<collection_name>.out_2.json`.
    """
    miner_path = shutil.which("lcm1")
    if miner_path is None:
        warnings.warn(f"Required mining program lcm1 not found in PATH.", UserWarning)
        raise RuntimeError(
            f"lcm1 is required but not installed or not in PATH. "
            f"Please install and compile it before running this package."
        )
    
    # Step 1: translate reactions into (item)sets of integers
    species2id = {}
    hyperedge2id = {}
    with open(collection_name+'.trns', 'w') as out:
        out.write("")

    transactions_1 = []
    transactions_2 = []

    for filename in os.listdir(rct_folder1):
        if filename[-5:] != '.json': continue
        with open(os.path.join(rct_folder1,filename)) as json_data:
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

            if hyperedge2id[hyperedge] not in transaction:
                transaction.append(hyperedge2id[hyperedge])
        
        transaction.sort()
        transactions_1.append(transaction)
        with open(collection_name+'.trns', 'a') as out:
            for item in transaction:
                out.write(str(item)+" ")
            out.write(" \n")

    for filename in os.listdir(rct_folder2):
        if filename[-5:] != '.json': continue
        with open(os.path.join(rct_folder2,filename)) as json_data:
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

            if hyperedge2id[hyperedge] not in transaction:
                transaction.append(hyperedge2id[hyperedge])
        
        transaction.sort()
        transactions_2.append(transaction)
        with open(collection_name+'.trns', 'a') as out:
            for item in transaction:
                out.write(str(item)+" ")
            out.write(" \n")

    num_1 = len(transactions_1)
    num_2 = len(transactions_2)
    print(num_1, num_2)

    # Step 2: use a frequent itemset miner to obtain patterns
    # https://research.nii.ac.jp/~uno/code/lcm.html
    frequency = int(frequency)
    os.system(f"lcm1 {collection_name}.trns {frequency}  > {collection_name}.frq")

    # Step 3: translate back the patterns to reactions
    id2hyperedge = {v: k for k, v in hyperedge2id.items()}

    def is_subset(smaller, larger):
        """Check if smaller is a subset of larger."""
        it = iter(larger)
        return all(elem in it for elem in smaller)

    itemsets = []
    frequencies = []
    frequencies_pair = []
    hits_pair = []
    assert len(transactions_1) == num_1 
    assert len(transactions_2) == num_2

    # get raw frequent hypergraph patterns 
    with open(collection_name+'.frq', 'r') as file:
        for line in file:
            toks = line.rstrip().split('(')
            freq = int(toks[1][:-1])

            itemset = toks[0].strip().split(' ')
            itemset = sorted( [ int(itm) for itm in itemset if (len(itm)>0)] )
            if len(itemset) == 0: continue 

            freq1 = 0
            hits_1 = []
            for i, trans in enumerate(transactions_1):
                if is_subset(itemset, trans):
                    freq1 += 1
                    hits_1.append(i)
            freq2 = 0
            hits_2 = []
            for i, trans in enumerate(transactions_2):
                if is_subset(itemset, trans):
                    freq2 += 1
                    hits_2.append(i)

            assert freq1+freq2 == freq
            itemsets.append(itemset)
            frequencies.append(freq)
            frequencies_pair.append( (freq1, freq2) )
            hits_pair.append( (hits_1, hits_2) )


    # filter out disconnected ones
    connected = []
    frequencies_conn = []
    graphs_conn = []

    for i in range(len(itemsets)):
        itemset = itemsets[i] # this is one pattern
        itemset = [ id2hyperedge[itm] for itm in itemset ]
        G = nx.Graph()
        G.graph['frequency'] = frequencies_pair[i]
        G.graph['hits'] = hits_pair[i]
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
            frequencies_conn.append(frequencies_pair[i])
        

    # filter more to keep just closed itemsets
    freq_bins = defaultdict(list)
    for i in range(len(connected)):
        freq_bins[frequencies_conn[i]].append(i)

    closed_indexes = []  
    freqs = list(freq_bins.keys())
    freqs.sort(key=lambda x: -x[0]-x[1])

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

    output_1 = []
    output_2 = []
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
        
        
        output_1.append( (hyperedges_out, graphs_conn[idx].graph['frequency'][0]) )
        output_2.append( (hyperedges_out, graphs_conn[idx].graph['frequency'][1]) )


    # Step 4: output patterns, remove temp files
    with open(collection_name+'.out_1.json', 'w') as out:
        json.dump(output_1, out)
    with open(collection_name+'.out_2.json', 'w') as out:
        json.dump(output_2, out)

    os.system(f'rm {collection_name}.trns {collection_name}.frq')

    return

def find_significant_patterns(
    pattern_file_pos: str, 
    pattern_file_neg: str, 
    no_sim_pos: int,
    no_sim_neg: int,
    alpha: float,
    use_fdr: bool,
    bestsup: int = 0
) -> None:

    pattern2freqs = {}

    num_trans_1 = int(no_sim_pos) # number positives n1
    num_trans_2 = int(no_sim_neg) # number negatives n0

    n1 = num_trans_1
    n = num_trans_1+num_trans_2
    use_fdr = ast.literal_eval(use_fdr)
    print('Use FDR:', use_fdr)

    # first dataset: positives
    with open(pattern_file_pos, 'r') as file: 
        data = json.load(file)

    for pattern in data:
        freq = pattern[1] # frequency
        pattern[0].sort()
        hypergraph = str(pattern[0])
        if hypergraph not in pattern2freqs:
            pattern2freqs[hypergraph] = [0,0]
        pattern2freqs[hypergraph][0] = freq

    # second dataset: negatives
    with open(pattern_file_neg, 'r') as file:
        data = json.load(file)

    for pattern in data:
        freq = pattern[1] # frequency
        pattern[0].sort()
        hypergraph = str(pattern[0])
        if hypergraph not in pattern2freqs:
            pattern2freqs[hypergraph] = [0,0]
        pattern2freqs[hypergraph][1] = freq


    bestsup = int(bestsup) # filtering by support
    if bestsup == 0:
        for minsup in range(50, 0, -1): # disregard patterns with supp < minsup    
            cnt = 0
            for patt in pattern2freqs:
                frqs = pattern2freqs[patt]
                if frqs[0] + frqs[1] >= minsup: cnt += 1
            
            nmin = min(n1, n-n1)
            # minumum att p-value for fisher test at minusp = r is
            # phi(r) =  (nmin choose r) / (n choose r) = (nmin! (n-r)!) / (n! (nmin-r)!)  if r < nmin
            #           1 / (n choose n1)
            if minsup < nmin: log_phi = lgamma(nmin+1) + lgamma(n-minsup+1) - lgamma(n+1) - lgamma(nmin-minsup+1)
            else:           log_phi = lgamma(nmin+1) + lgamma(n-nmin+1) - lgamma(n+1)
            
            if use_fdr: # Benjamini-Yekutieli 
                if log(log(cnt+1)+1) + log_phi > log(alpha):
                    bestsup = minsup+1
                    break
            else: # use FWER: Tarone
                if log(cnt+1) + log_phi > log(alpha):
                    bestsup = minsup+1
                    break

    print("Minimum (combined) frequency:", bestsup)

    # get patterns that could be significant
    pattern_list = []
    pval_list = []
    effect_list = []
    freqs_list = []

    for pattern in pattern2freqs:
        if pattern2freqs[pattern][0] + pattern2freqs[pattern][1] < bestsup:
            continue # discarded 
        
        pattern_list.append(pattern)

        table =[[ pattern2freqs[pattern][0] , n1 - pattern2freqs[pattern][0]],  
                [ pattern2freqs[pattern][1] , n-n1-pattern2freqs[pattern][1]]]
        
        pval = fisher_exact(table, alternative='two-sided').pvalue
        odds = odds_ratio(table, kind='conditional').statistic

        pval_list.append(pval)
        effect_list.append(odds)
        freqs_list.append([pattern2freqs[pattern][0], pattern2freqs[pattern][1]])

    print("Num patterns:", len(pval_list))

    if use_fdr:
        sign, qvals, _, _ = statsmodels.stats.multitest.multipletests(pval_list, alpha=alpha, method='fdr_by', is_sorted=False, returnsorted=False)
    else:
        sign, qvals, _, _ = statsmodels.stats.multitest.multipletests(pval_list, alpha=alpha, method='bonferroni', is_sorted=False, returnsorted=False)


    out_list = [[eval(pattern), freqs_list[i][0], freqs_list[i][1], bool(sign[i])] for i, pattern in enumerate(pattern_list)]

    with open(pattern_file_pos.split('.')[0]+'.sign.json', 'w') as out:
        json.dump(out_list, out)

    print("Significant patterns")
    for i in range(len(pattern_list)):
        if not sign[i]: continue
        print(pattern_list[i])
        print('Positive/negative freq:', freqs_list[i][0], freqs_list[i][1])
        print('P-value and odds ratio:', pval_list[i], effect_list[i])
        print()
    
