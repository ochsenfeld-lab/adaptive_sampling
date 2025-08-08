# Reaction Hypergraphs 

This code is an implementation of the algorithms described in the paper "*Automated Discovery of Reactive Events via Hypergraph Mining of Ab Initio Atomistic Simulations*" by Alexandra Stan-Bernhardt, Paolo Pellizzoni, Karsten Borgwardt, and Christian Ochsenfeld.

### Requirements

First compile the frequent itemset miner (miners/lcm1.c) with a c/c++ compiler: ```gcc -O3 -o lcm1 lcm1.c```.

The code is written in Python, and it also requires the packages ```networkx```, ```numpy```, ```scipy``` and ```statsmodels```.

The programs expect data to be stored in a folder (one per experimental condition), with an individual JSON file for each simulation. 

### Frequent patterns

One can extract the frequent reactive pathways using ```find_frequent_patterns.py```. The first argument is the data folder, the second argument is the name for the output file, and the third is the frequency threshold. An example usage is the following:

```python find_frequent_patterns.py data/reactions_lists_39_nowater/ 39_nowater 10```

This outputs a file ```39_nowater.out.json``` which can be read, e.g, using ```python utils/print_patterns.py 39_nowater.out.json```.

### Statistically significant patterns

One can extract the reactive pathways that are statistically significantly enriched in one of two experimental conditions as follows.
First, one needs to extract all reactive pathways with a minimum frequency across both experimental conditions using ```find_frequent_patterns_pair.py```. The first and second arguments are the data folders for the two conditions, the third argument is the name for the output file, and the fourth is the frequency threshold. Here, the frequency threshold is introduced mostly for computational complexity reasons, to reduce the number of patterns, and should be set as low as possible. An example usage is the following:

```python find_frequent_patterns_pair.py data/reactions_lists_39_nowater/ data/reactions_lists_62_nowater/ nowater 4```

This outputs two files (one per condition) ```nowater.out_1.json``` and ```nowater.out_2.json```. These can also be read using ```python utils/print_patterns.py```. 

Once the patterns have been extracted, the statistical analysis can be performed using ```find_significant_patterns.py```. The first and second argument are the JSON files with the pattern frequencies, as outputted by the previous program. The third and fourth arguments are the number of simulations in the two conditions. The fifth argument is the level α at which one wants to control the FWER/FDR, and the sixth argument is whether to use FDR control (i.e., False for FWER and True for FDR). Finally, the seventh argument is the minimum frequency (across both experimental conditions) under which to discard patterns; if left to 0, this value is computed automatically using Tarone's method. An example usage is the following:

```python find_significant_patterns.py nowater.out_1.json nowater.out_2.json 97 97 0.1 False 0```

This outputs a file ```nowater.sign.json``` which can be read, e.g, using ```python utils/print_patterns.py nowater.sign.json```.