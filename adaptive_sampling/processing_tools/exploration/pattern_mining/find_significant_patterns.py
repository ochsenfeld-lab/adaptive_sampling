import sys, os
from math import log, lgamma
import json 
from collections import defaultdict
from scipy.stats import fisher_exact
from scipy.stats.contingency import odds_ratio
import statsmodels.stats.multitest
import ast
import numpy as np

pattern2freqs = {}

if(len(sys.argv) != 8):
    raise Exception("Usage: positives_file negatives_file n1 n0 alpha use_fdr")

num_trans_1 = int(sys.argv[3]) # number positives n1
num_trans_2 = int(sys.argv[4]) # number negatives n0

n1 = num_trans_1
n = num_trans_1+num_trans_2
alpha = float(sys.argv[5])
use_fdr = ast.literal_eval(sys.argv[6])
print('Use FDR:', use_fdr)

# first dataset: positives
with open(sys.argv[1], 'r') as file: 
    data = json.load(file)

for pattern in data:
    freq = pattern[1] # frequency
    pattern[0].sort()
    hypergraph = str(pattern[0])
    if hypergraph not in pattern2freqs:
        pattern2freqs[hypergraph] = [0,0]
    pattern2freqs[hypergraph][0] = freq

# second dataset: negatives
with open(sys.argv[2], 'r') as file:
    data = json.load(file)

for pattern in data:
    freq = pattern[1] # frequency
    pattern[0].sort()
    hypergraph = str(pattern[0])
    if hypergraph not in pattern2freqs:
        pattern2freqs[hypergraph] = [0,0]
    pattern2freqs[hypergraph][1] = freq


bestsup = int(sys.argv[7]) # filtering by support
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

with open(sys.argv[1].split('.')[0]+'.sign.json', 'w') as out:
    json.dump(out_list, out)

print("Significant patterns")
for i in range(len(pattern_list)):
    if not sign[i]: continue
    print(pattern_list[i])
    print('Positive/negative freq:', freqs_list[i][0], freqs_list[i][1])
    print('P-value and odds ratio:', pval_list[i], effect_list[i])
    print()
    
