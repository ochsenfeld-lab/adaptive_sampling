import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
from pyvis.network import Network

from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.Draw import IPythonConsole

from .nanoreactor_network import *

def generate_mol_grid(df: pd.DataFrame):
    ''' Generate molecule grid with all molecules that have an absolute occurence higher than 10 in the nanoreactor simulation.

    Args:
        df: data frame from molecular nanoreactor simulation
    Returns:
        -
    '''

    abs_prob = {}

    for timestep in range(len(df['Time step [fs]'])):
        for struc in df['SMILES'][timestep]:
            for elem in struc:
                if elem in abs_prob:
                    abs_prob[elem] += 1
                else:
                    abs_prob[elem] = 1

    filtered_abs_prob = dict(filter(lambda elem: elem[1] > 10, abs_prob.items()))
    sorted_abs_prob = (sorted(filtered_abs_prob, key=filtered_abs_prob.get, reverse=True))

    list_mols = []
    list_formulas = []

    for i in range(len(sorted_abs_prob)):
        if sorted_abs_prob[i] != "Revise structure":
            mol=Chem.MolFromSmiles(sorted_abs_prob[i],sanitize=False)

            Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|
                            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
                            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                            catchErrors=True)
            list_formulas.append(CalcMolFormula(mol) + ": " + str(abs_prob[sorted_abs_prob[i]]))
            list_mols.append(mol)

    print("There are " + str(len(list_mols)) + " distinct molecules on the grid.\n")
    img = Chem.Draw.MolsToGridImage(list_mols, molsPerRow=7, legends = list_formulas, subImgSize=[300,300], maxMols = 999999, returnPNG = False)
    img.save("sim_grid.png", dpi = (1500.0,1500.0))

    return

def generate_network_grid(nanoNet: NanoNetwork):
    ''' Plot molecule grid as a network legend.

    Args:
        nanoNet: NanoNetwork object
    Returns:
        -
    '''

    u_list=[]
    for u in nanoNet.graph.nodes():
        if len(nanoNet.graph.in_edges(u)) >= 1 or len(nanoNet.graph.out_edges(u)) >= 1:
            u_list.append(u)

    list_formulas=[]
    list_mols=[]

    for i in u_list:
        if nanoNet.node_labels[i] != "Revise structure":

            mol = Chem.MolFromSmiles(nanoNet.node_labels[i], sanitize=False)
            Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|
                            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
                            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                            catchErrors=True)

            list_formulas.append(str(i) + ": "+ CalcMolFormula(mol))
            list_mols.append(mol)
    img = Chem.Draw.MolsToGridImage(list_mols, molsPerRow=7, legends = list_formulas, subImgSize=[300,300], maxMols = 999999, returnPNG=False)
    img.save("net_grid.png", dpi = (1500.0,1500.0))
    
    return

def generate_bar_plot(df: pd.DataFrame):
    ''' Generate bar plot to get a quick overview over the size of the molecular species found during the simulation.

    Args:
        df: data frame from molecular nanoreactor simulation
    Returns:
        -
    '''

    molecular_formulas = df['Molecular Formulas'] 

    dictionary = {}
    keys_list=[]

    for ts in range(0,len(df['Time step [fs]']),1):
        for key in molecular_formulas[ts]:
            if key not in dictionary:
                dictionary[key] = {}
                keys_list.append(key)

    for key in keys_list:
        for ts in range(0,len(molecular_formulas),1):
            dictionary[key][df['Time step [fs]'][ts]] = 0


    for ts in range(0,len(molecular_formulas),1):
        for key in molecular_formulas[ts]:
            dictionary[key][df['Time step [fs]'][ts]] += 1

    # Area plot fragments against time

    df = pd.DataFrame(dictionary)

    sums= {}
    for i in df.columns:
          sums[i]=df[i].sum()

    sums = pd.DataFrame(sums, index=["sum"])
    df = pd.concat([df.loc[:],sums])

    sorted_df = df.sort_values(df.last_valid_index(), axis=1, ascending = False) # type: ignore

    f=sorted_df.loc["sum"]>10
    sorted_df = sorted_df[sorted_df.columns[f]]

    sorted_df[:-1].plot(kind="area", stacked=True, cmap=cm.tab20b, figsize = (15,7), linewidth = 0)
    plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol = 4)
    plt.locator_params(axis="x", nbins=10)

    plt.ylabel("# Fragments")
    plt.xlabel("Time [fs]")

    plt.savefig('bar_plot.png',dpi = 300, bbox_inches = "tight")

    return

def plot_static_network_kk(nanoNet: NanoNetwork):
    ''' Plot network in kamada_kawai layout.

    Args:
        nanoNet: NanoNetwork object
    Returns:
        -
    '''

    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)

    # draw network
    nc = nx.draw(nanoNet.graph, alpha=0.8, pos=nx.kamada_kawai_layout(nanoNet.graph), with_labels=True, linewidths=1.0, node_size=150, node_color=nanoNet.node_colors, edge_color=nanoNet.edge_colors, width=1,font_size=10, arrowsize=8)
    plt.savefig('network_kk.png', dpi=300, bbox_inches='tight')
    
    return

def plot_static_network_shell(nanoNet: NanoNetwork):
    ''' Plot network in shell layout.

    Args:
        nanoNet: NanoNetwork object
    Returns:
        -
    '''

    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)

    # draw network
    nc = nx.draw(nanoNet.graph, alpha=0.8, pos=nx.shell_layout(nanoNet.graph), with_labels=True, linewidths=1.0, node_size=150, node_color=nanoNet.node_colors, edge_color=nanoNet.edge_colors, width=1,font_size=10, arrowsize=8)
    plt.savefig('network_shell.png', dpi=300, bbox_inches='tight')
    
    return

def plot_interactive_network(nanoNet: NanoNetwork):
    ''' Obtain html network plot using Pyvis.

    Args:
        nanoNet: NanoNetwork object 
    Returns:
        -
    '''

    net = Network(directed = True)
    net.from_nx(nanoNet.graph)
    net.show('nodes.html')