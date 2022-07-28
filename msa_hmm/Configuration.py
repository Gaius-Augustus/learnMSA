#the configuration can be changed by experienced users
#proper command line support for these parameters will be added in the future
default = {
    
    "transition_init" : "default",
    "flank_init" : "default",
    "emission_init" : "background",
    "alpha_flank" : 7000,
    "alpha_single" : 1e9,
    "alpha_frag" : 1e4,
    "max_surgery_runs" : 4,
    "length_init_quantile" : 0.5,
    "surgery_quantile" : 0.5,
    "min_surgery_seqs" : 1e4,
    "len_mul" : 0.8,
    "batch_size" : "adaptive",
    "use_prior" : True,
    "dirichlet_mix_comp_count" : 1,
    "use_anc_probs" : True,
    "tau_init" : 0.0,
    "keep_tau" : False
    
}

def as_str(config, items_per_line=3):
    return " , ".join(key + " : " + str(val) + "\n"*((i+1)%items_per_line==0) for i,(key,val) in enumerate(config.items()))