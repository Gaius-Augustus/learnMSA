import logomaker
from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import tensorflow as tf
import msa_hmm


def make_logo(alignment, ax):
    hmm_cell = alignment.msa_hmm_layer.C
    
    logomaker_alphabet = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    logomaker_perm = np.array([msa_hmm.fasta.alphabet.index(aa) for aa in logomaker_alphabet], dtype=int)

    #reduce to std AA alphabet 
    emissions = hmm_cell.make_B().numpy()[1:hmm_cell.length+1,:20][:,logomaker_perm]

    information_content = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)(
                                                         emissions,
                                                         np.expand_dims(msa_hmm.ut.background_distribution[:20], 0))
    information_content = tf.expand_dims(information_content, -1).numpy()

    information_content_df = pd.DataFrame(information_content * emissions, 
                                          columns=logomaker_alphabet)

    ax.figure.set_size_inches(hmm_cell.length/2, 6)

    # create Logo object
    logo = logomaker.Logo(information_content_df,
                       color_scheme='skylign_protein',
                       vpad=.1,
                       width=.8,
                       ax=ax)

    # style using Axes methods
    logo.ax.set_ylabel('information content')
    logo.ax.set_xlim([-1, len(information_content_df)])
    

    
    
def plot_hmm(alignment, 
             ax,
               edge_show_threshold=1e-2, #grey out edges below this threshold (label_probs=False requires adjustment)
               num_aa=3, #print this many of the aminoacids with highest emission probability per match state
               label_probs=True, #label edges with probabilities, label with kernel values otherwise
               seq_i=-1, #index of the sequence, which viterbi sequence is plotted, -1 means no sequence 
               spacing=1.0,
               path_color="#CC6600",
               active_transition_color="#000000",
               inactive_transition_color="#E0E0E0"
                ):
    hmm_cell = alignment.msa_hmm_layer.C
    
    G = nx.DiGraph()
    indices_dict = hmm_cell.sparse_transition_indices_explicit()
    probs = hmm_cell.make_probs()
    for transition_type, indices in indices_dict.items():
        if transition_type == "begin_to_match" or transition_type == "match_to_end":
            continue
        type_probs = probs[transition_type]
        type_colors = [active_transition_color if p > edge_show_threshold else inactive_transition_color for i,p in zip(indices, type_probs)]
        for (u,v), c in zip(indices, type_colors):
            G.add_edge(u, v, color=c, label=transition_type)
    pos = {}
    pos[0] = (-spacing*2, spacing) # left flanking state
    for i in range(1, hmm_cell.length+1): #match-states
        pos[i] = (spacing*i, spacing/2)
    for i in range(hmm_cell.length-1): #insertions
        pos[hmm_cell.length+1+i] = (1.75*spacing + spacing*i, -spacing/2)
    pos[2*hmm_cell.length] = (spacing*(hmm_cell.length+2) / 2, 1.5*spacing) #unannotated-state
    pos[2*hmm_cell.length+1] = (spacing*(hmm_cell.length+6), spacing) #right flank state
    pos[2*hmm_cell.length+2] = (spacing*(hmm_cell.length+7), spacing) #terminal state
    pos[2*hmm_cell.length+3] = (-spacing*1.5, spacing/1.5) #begin state
    pos[2*hmm_cell.length+4] = (spacing*(hmm_cell.length+5), spacing/1.5) #end state
    for i in range(hmm_cell.length): #deletions
        pos[2*hmm_cell.length+5+i] = (spacing*(i+0.9), -1.5*spacing)

    edge_labels = {}
    if label_probs:
        values = probs
    else:
        values = {part_name : kernel.numpy() for part_name, kernel in hmm_cell.transition_kernel.items()}
    for transition_type, indices in indices_dict.items():
        if transition_type == "begin_to_match" or transition_type == "match_to_end":
            continue
        edge_labels.update({(edge[0], edge[1]) : "%.2f" % v 
                                for edge,p,v in zip(indices, probs[transition_type], values[transition_type]) 
                                    if p > edge_show_threshold})

    B = hmm_cell.make_B().numpy()
    node_labels = {}
    for i in range(hmm_cell.length):
        sort = np.argsort(-B[i+1, :25])
        label = str(i+1)+"\n"
        for j in range(num_aa):
            label += msa_hmm.fasta.alphabet[sort[j]] + " (" + "%.2f" % B[i+1, sort[j]] + ")\n"
        label += "en="+"%.2f" % probs["begin_to_match"][i]+"\n"
        label += "ex="+"%.2f" % probs["match_to_end"][i]+"\n"
        node_labels[i+1] = label

    ax.figure.set_size_inches(hmm_cell.length, 7)
    
    p = tf.math.sigmoid(hmm_cell.flank_init)
    ax.text(-4*spacing, 0, "Init: \n (%.2f, %.2f)" % (p, 1-p), fontsize=12)
    ax.text(-1.5*spacing, -spacing/2, "Insertions", fontsize=20)
    ax.text(-1.5*spacing, -1.5*spacing, "Deletions", fontsize=20)
    edge_colors = [G[u][v]['color'] for u,v in G.edges()]
    nx.draw_networkx_edges(G, pos, 
                            node_size=10, 
                            arrowstyle="fancy", 
                            width=1, 
                            edge_color=edge_colors, 
                            ax=ax) 
    nx.draw_networkx_edge_labels(G, 
                                pos, 
                                edge_labels=edge_labels, 
                                label_pos=0.6,
                                font_size=10, 
                                ax=ax)
    label_pos = {i : (x, y+0.1+0.05*num_aa) for i, (x,y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, labels=node_labels, font_size=8)

    if seq_i != -1:
        sequence = alignment.fasta_file.get_raw_seq(alignment.indices[seq_i])
        sequence = np.append(sequence, msa_hmm.fasta.s) #append terminal symbol
        sequence = np.expand_dims(sequence, 0)
        hidden_seq = msa_hmm.align.viterbi(sequence, hmm_cell)
        hidden_seq = list(hidden_seq[0])
        for i in range(len(hidden_seq)):
            #find silent path parts along delete states
            #match to match
            if (i < len(hidden_seq)-1 and
                hidden_seq[i] > 0 and 
                hidden_seq[i] < hmm_cell.length+1 and
                hidden_seq[i+1] > 0 and 
                hidden_seq[i+1] < hmm_cell.length+1):
                for j in range(hidden_seq[i+1]-1-hidden_seq[i]):
                    hidden_seq.insert(i+1+j, 2*hmm_cell.length+5+hidden_seq[i]+j)
        i = 0
        while hidden_seq[i]==0:
            i+=1
        hidden_seq.insert(i, 2*hmm_cell.length+3)
        i = 1
        while hidden_seq[-i]>=2*hmm_cell.length+1:
            i+=1
        hidden_seq.insert(-i+1, 2*hmm_cell.length+4)

        edgelist = list(zip(hidden_seq[:-1], hidden_seq[1:]))
        edge_labels_path = {e : l for e,l in edge_labels.items() if e in edgelist}
        nx.draw_networkx_edges(G, pos, 
                                edgelist=edgelist, 
                                edge_color = path_color, 
                                node_size=10, 
                                arrowstyle="fancy", 
                                width=4, 
                                ax=ax)
        nx.draw_networkx_edge_labels(G, 
                                    pos, 
                                    edge_labels=edge_labels_path, 
                                    label_pos=0.6,
                                    font_size=10, 
                                    font_color=path_color,
                                    ax=ax)
    plt.subplots_adjust(left=0.4, right=0.6, top=0.9, bottom=0.1)