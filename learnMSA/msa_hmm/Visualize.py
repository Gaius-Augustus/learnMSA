import logomaker
from matplotlib import pyplot as plt
from matplotlib.patches import ArrowStyle
import networkx as nx
import pandas as pd
import numpy as np
import tensorflow as tf
from learnMSA import msa_hmm
import itertools
import seaborn as sns


def plot_logo(am, model_index, ax):
    hmm_cell = am.msa_hmm_layer.cell
    hmm_cell.recurrent_init()
    length = hmm_cell.length[model_index]
    
    logomaker_alphabet = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    logomaker_perm = np.array([msa_hmm.fasta.alphabet.index(aa) for aa in logomaker_alphabet], dtype=int)

    #reduce to std AA alphabet 
    emissions = hmm_cell.emitter[0].make_B_amino().numpy()[model_index, 1:length+1,:20][:,logomaker_perm]
    background = hmm_cell.emitter[0].make_B_amino().numpy()[model_index, 0, :20][logomaker_perm]
    #background = hmm_cell.emitter[0].emission_init[model_index]((length,25), dtype=am.msa_hmm_layer.dtype)[0, :20]
    information_content = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)(
                                                         emissions,
                                                         np.expand_dims(background, 0))
    information_content = tf.expand_dims(information_content, -1).numpy()

    information_content_df = pd.DataFrame(information_content * emissions, 
                                          columns=logomaker_alphabet)

    ax.figure.set_size_inches(length/2, 6)

    # create Logo object
    logo = logomaker.Logo(information_content_df,
                       color_scheme='skylign_protein',
                       vpad=.1,
                       width=.8,
                       ax=ax)

    # style using Axes methods
    logo.ax.set_ylabel('information content')
    logo.ax.set_xlim([-1, len(information_content_df)])
    
    
def plot_hmm(am, 
             model_index,
             ax,
               edge_show_threshold=1e-2, #grey out edges below this threshold (label_probs=False requires adjustment)
               num_aa=3, #print this many of the aminoacids with highest emission probability per match state
               label_probs=True, #label edges with probabilities, label with kernel values otherwise
               seq_indices=[], #index of the sequence, which viterbi sequence is plotted, -1 means no sequence 
               spacing=1.0,
               path_colors=[],
               active_transition_color="#000000",
               inactive_transition_color="#E0E0E0"
                ):
    hmm_cell = am.msa_hmm_layer.cell
    hmm_cell.recurrent_init()
    length = hmm_cell.length[model_index]
    
    G = nx.DiGraph()
    indices_dict = hmm_cell.transitioner.sparse_transition_indices_explicit[model_index]
    probs = hmm_cell.transitioner.make_probs()[model_index]
    for transition_type, indices in indices_dict.items():
        if transition_type == "begin_to_match" or transition_type == "match_to_end":
            continue
        type_probs = probs[transition_type]
        type_colors = [active_transition_color if p > edge_show_threshold else inactive_transition_color for i,p in zip(indices, type_probs)]
        for (u,v), c in zip(indices, type_colors):
            G.add_edge(u, v, color=c, label=transition_type)
    pos = {}
    pos[0] = (-spacing*2, spacing) # left flanking state
    for i in range(1, length+1): #match-states
        pos[i] = (spacing*i, spacing/2)
    for i in range(length-1): #insertions
        pos[length+1+i] = (1.75*spacing + spacing*i, -spacing/2)
    pos[2*length] = (spacing*(length+2) / 2, 1.5*spacing) #unannotated-state
    pos[2*length+1] = (spacing*(length+6), spacing) #right flank state
    pos[2*length+2] = (spacing*(length+7), spacing) #terminal state
    pos[2*length+3] = (-spacing*1.5, spacing*1.5) #begin state
    pos[2*length+4] = (spacing*(length+5), spacing*1.5) #end state
    for i in range(length): #deletions
        pos[2*length+5+i] = (spacing*(i+0.9), -1.5*spacing)

    edge_labels = {}
    if label_probs:
        values = probs
    else:
        values = {part_name : kernel.numpy() for part_name, kernel in hmm_cell.transitioner.transition_kernel[model_index].items()}
    for transition_type, indices in indices_dict.items():
        if transition_type == "begin_to_match" or transition_type == "match_to_end":
            continue
        edge_labels.update({(edge[0], edge[1]) : "%.2f" % v 
                                for edge,p,v in zip(indices, probs[transition_type], values[transition_type]) 
                                    if p > edge_show_threshold})

    B = hmm_cell.emitter[0].make_B_amino().numpy()[model_index]
    node_labels = {}
    for i in range(length):
        sort = np.argsort(-B[i+1, :25])
        label = str(i+1)+"\n"
        for j in range(num_aa):
            label += msa_hmm.fasta.alphabet[sort[j]] + " (" + "%.2f" % B[i+1, sort[j]] + ")\n"
        label += "en="+"%.2f" % probs["begin_to_match"][i]+"\n"
        label += "ex="+"%.2f" % probs["match_to_end"][i]+"\n"
        node_labels[i+1] = label

    ax.figure.set_size_inches(length, 7)
    
    p = tf.math.sigmoid(hmm_cell.transitioner.flank_init_kernel[model_index])
    #ax.text(-4*spacing, 0, "Init: \n (%.2f, %.2f)" % (p, 1-p), fontsize=12)
    ax.text(-2.5*spacing, -spacing/2, "Insertions", fontsize=20)
    ax.text(-2.5*spacing, -1.5*spacing, "Deletions", fontsize=20)
    edge_colors = [G[u][v]['color'] for u,v in G.edges()]
    nx.draw_networkx_edges(G, pos, 
                            node_size=10, 
                            width=1, 
                            edge_color=edge_colors, 
                            ax=ax) 
    edge_label_pos = {n : (x,y-0.09) for n,(x,y) in pos.items()}
    nx.draw_networkx_edge_labels(G, 
                                edge_label_pos, 
                                edge_labels=edge_labels, 
                                label_pos=0.6,
                                font_size=10, 
                                ax=ax)
    label_pos = {i : (x, y+0.1+0.05*num_aa) for i, (x,y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, labels=node_labels, font_size=8)
    
    for k, (seq_i, path_color) in enumerate(zip(seq_indices, path_colors)):
        ds = msa_hmm.train.make_dataset(np.array([seq_i]), am.batch_generator, batch_size=1, shuffle=False)
        for x, _ in ds:
            sequence = am.encoder_model(x)  
        hidden_seq = msa_hmm.viterbi.viterbi(sequence, hmm_cell).numpy()[model_index]
        hidden_seq = list(hidden_seq[0])
        for i in range(len(hidden_seq)):
            #find silent path parts along delete states
            #match to match
            if (i < len(hidden_seq)-1 and
                hidden_seq[i] > 0 and 
                hidden_seq[i] < length+1 and
                hidden_seq[i+1] > 0 and 
                hidden_seq[i+1] < length+1):
                for j in range(hidden_seq[i+1]-1-hidden_seq[i]):
                    hidden_seq.insert(i+1+j, 2*length+5+hidden_seq[i]+j)
            #find unannotated segment to match transitions
            if (i < len(hidden_seq)-1 and
               hidden_seq[i+1] > 0 and 
               hidden_seq[i+1] < length+1 and
               hidden_seq[i] == 2*length):
                hidden_seq.insert(i+1, 2*length+3)
            #find match to unannotated segment transitions
            if (i < len(hidden_seq)-1 and
               hidden_seq[i] > 0 and 
               hidden_seq[i] < length+1 and
               hidden_seq[i+1] == 2*length):
                hidden_seq.insert(i+1, 2*length+4)
                
        i = 0
        while hidden_seq[i]==0:
            i+=1
        hidden_seq.insert(i, 2*length+3)
        i = 1
        while hidden_seq[-i]>=2*length+1:
            i+=1
        hidden_seq.insert(-i+1, 2*length+4)

        edgelist = list(zip(hidden_seq[:-1], hidden_seq[1:]))
        edge_labels_path = {e : l for e,l in edge_labels.items() if e in edgelist}
        edge_pos = {n : (x+k*spacing*0.05,y+k*spacing*0.1) if n > length and n < 2*length else (x+k*spacing*0.05,y+k*spacing*0.05) for n,(x,y) in pos.items()}
        nx.draw_networkx_edges(G, edge_pos, 
                                edgelist=edgelist, 
                                edge_color = path_color, 
                                node_size=10, 
                                width=6, 
                                min_target_margin=5,
                                ax=ax)
        
    #make a legend for the hidden paths
    if len(seq_indices) > 0:
        f = lambda c: ax.plot([],[], color=c)[0]
        handles = [f(col) for col in path_colors]
        labels = [am.fasta_file.seq_ids[seq_i] for seq_i in seq_indices]
        leg = ax.legend(handles, labels, loc="lower right", prop={'size': 18})
        for line in leg.get_lines():
            line.set_linewidth(8.0)
    plt.subplots_adjust(left=0.4, right=0.6, top=0.9, bottom=0.1)
    
    
def plot_anc_probs(am, 
                   model_index,
                   seqs=[0,1,2], 
                   pos=list(range(6)), 
                   rescale=True, 
                   title="Site-wise ancestral probabilities"):
    n, m = len(seqs), len(pos)
    ds = msa_hmm.train.make_dataset(am.indices[seqs], 
                                    am.batch_generator,
                                    batch_size=n, 
                                    shuffle=False)
    for x,_ in ds:
        ancs = am.encoder_model(x).numpy()[model_index]
    i = [l.name for l in am.encoder_model.layers].index("anc_probs_layer")
    anc_probs_layer = am.encoder_model.layers[i]
    indices = np.stack([am.indices]*am.msa_hmm_layer.cell.num_models)
    indices = np.expand_dims(indices, -1)
    tau = anc_probs_layer.make_tau(indices)[model_index]
    if rescale:
        ancs /= np.sum(ancs, -1, keepdims=True)
    f, axes = plt.subplots(n, m, sharey=True)
    axes = axes.flatten()
    f.set_size_inches(3+3*m, 2*n)
    for a,(s,i) in enumerate(itertools.product(seqs, pos)):
        sns.barplot(x=msa_hmm.fasta.alphabet[:20], y=ancs[s,i,:20], ax=axes[a]);
        if a % m == 0:
            axes[a].annotate(f"tau={'%.3f'%tau[s]} ->", (0.3,0.9*axes[a].get_ylim()[1]))
    f.suptitle(title, fontsize=16)
    
    
def plot_rate_matrices(am,
                       model_index,
                       title="normalized rate matrix (1 time unit = 1 expected mutation per site)"):
    i = [l.name for l in am.encoder_model.layers].index("anc_probs_layer")
    anc_probs_layer = am.encoder_model.layers[i]
    Q = anc_probs_layer.make_Q()[model_index]
    k = Q.shape[0]
    f, axes = plt.subplots(1, k, sharey=True)
    f.set_size_inches(10*k, 10.5)
    if k > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    for i,ax in enumerate(axes):
        a = msa_hmm.fasta.alphabet[:20]
        sns.heatmap(Q[i], linewidth=0.5, ax=ax, xticklabels=a, yticklabels=a)
    f.suptitle(title, fontsize=16)
    
    
def print_and_plot(am, 
                   model_index = None,
                   max_seq = 20, 
                   seqs_to_plot = [0,1,2], 
                   seq_ids = False, 
                   show_model=True, 
                   show_anc_probs=True,
                   show_logo=True,
                   model_filename="", 
                   anc_probs_filename="",
                   logo_filename=""):
    if model_index is None:
        model_index = am.best_model
    # print the alignment
    msa = am.to_string(model_index)
    i = [l.name for l in am.encoder_model.layers].index("anc_probs_layer")
    anc_probs_layer = am.encoder_model.layers[i]
    ds = msa_hmm.train.make_dataset(am.indices, 
                            am.batch_generator,
                            am.batch_size, 
                            shuffle=False)
    ll = am.model.predict(ds)[:,model_index] + am.compute_log_prior()[model_index]
    for i,s in enumerate(msa[:max_seq]):
        indices = np.array([[am.indices[i]]]*am.msa_hmm_layer.cell.num_models)
        tau = anc_probs_layer.make_tau(indices)[model_index]
        param_string = "l=%.2f" % (ll[i]) + "_t=%.2f" % tau
        if seq_ids:
            print(f">{am.fasta_file.seq_ids[am.indices[i]]} "+param_string)
        else:
            print(">"+param_string)
        print(s)
    if len(msa) > max_seq:
        print(len(msa) - max_seq, "sequences omitted.")
    if show_model:
        #plot the model
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        plot_hmm(am, model_index, ax, 
                 seq_indices=am.indices[seqs_to_plot],
                 path_colors=["#CC6600", "#0000cc", "#00cccc"])   
        if model_filename != "":
            plt.savefig(model_filename, bbox_inches='tight')
    if show_anc_probs:
        plot_anc_probs(am, model_index, seqs=seqs_to_plot)
        if anc_probs_filename != "":
            plt.savefig(anc_probs_filename, bbox_inches='tight')
    if show_logo:
        fig, ax = plt.subplots()
        plot_logo(am, model_index, ax)
        if logo_filename != "":
            plt.savefig(logo_filename, bbox_inches='tight')
        
        
def plot_sequence_length_distribution(fasta_filename, 
                                     bins=100,
                                     q=0.75):
    fasta_file = msa_hmm.fasta.Fasta(fasta_filename)
    x = fasta_file.seq_lens
    sns.histplot(x, bins=bins)
    #plt.hist(x, density=False, bins=bins);  
    plt.xlabel("Seq. length")
    plt.ylabel("Number of seq.")
    median, q25, q75= np.percentile(x, 50), np.percentile(x, 25), np.percentile(x, 75)
    plt.axvline(x = q25, c='k', ls='--', label = "q25")
    plt.axvline(x = median, c='orange', ls='-', label = "median")
    plt.axvline(x = q75, c='g', ls='--', label = "q75")
    plt.legend(loc='upper right')