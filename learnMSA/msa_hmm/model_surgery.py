from attr import dataclass
import numpy as np
import tensorflow as tf

import learnMSA.msa_hmm.Initializers as initializers
from learnMSA.msa_hmm.AlignmentModel import AlignmentModel


def get_discard_or_expand_positions(am, del_t=0.5, ins_t=0.5):
    """ Given an AlignmentModel, computes positions for match expansions and discards based on the posterior state probabilities.
    Args:
        am: An AlignmentModel object.
        del_t: Discards match positions that are expected less often than this number.
        ins_t: Expands insertions that are expected more often than this number.
                Adds new match states according to the expected insertion length.
    Returns:
        pos_expand: A list of arrays with match positions to expand per model.
        expansion_lens: A list of arrays with the expansion lengths.
        pos_discard: A list of arrays with match positions to discard.
    """
    # num_models x max_num_states
    expected_state = am.model.posterior(am.indices, am.batch_size)
    pos_expand = []
    expansion_lens = []
    pos_discard = []
    for i in range(am.num_models):
        model_length = am.msa_hmm_layer.cell.length[i]
        #discards
        match_states = expected_state[i, 1:model_length+1]
        discard = np.arange(model_length, dtype=np.int32)[match_states < del_t]
        pos_discard.append(discard)
        #expansions
        insert_states = expected_state[i, model_length+1:2*model_length]
        left_flank_state = expected_state[i, 0]
        right_flank_state = expected_state[i, 2*model_length+1]
        all_inserts = np.concatenate([[left_flank_state], insert_states, [right_flank_state]], axis=0)
        which_to_expand = all_inserts > ins_t
        expand = np.arange(model_length+1, dtype=np.int32)[which_to_expand]
        pos_expand.append(expand)
        #expansion lengths
        expand_len = np.ceil(all_inserts).astype(np.int32)[which_to_expand]
        expansion_lens.append(expand_len)
    return pos_expand, expansion_lens, pos_discard


#applies discards and expansions simultaneously to a vector x
#all positions are with respect to the original vector without any modification
#replicates insert_value for the expansions
#assumes that del_marker is a value that does no occur in x
#returns a new vector with all modifications applied
def apply_mods(x, pos_expand, expansion_lens, pos_discard, insert_value, del_marker=-9999):
    #mark discard positions with del_marker, expand thereafter
    #and eventually remove the marked positions
    x = np.copy(x)
    x[pos_discard] = del_marker
    rep_expand_pos = np.repeat(pos_expand, expansion_lens)
    x = np.insert(x, rep_expand_pos, insert_value, axis=0)
    if len(x.shape) == 2:
        x = x[np.any(x != del_marker, -1)]
    else:
        x = x[x != del_marker]
    return x


# makes updated pos_expand, expansion_lens, pos_discard vectors that fulfill:
#
# - each consecutive segment of discards from i to j is replaced with discards
#   from i+k-1 to j+k and an expansion of length 1 at i+k-1
#   edge cases that do not require an expansion:
#        replaced with discards from i+k to j+k if i+k == 0 and j+k < L-1
#        replaced with discards from i+k-1 to j+k-1 if i+k > 0 and j+k == L-1
#        replaced with discards from i+k to j+k-1 i+k == 0 and j+k == L-1
#
# - an expansion at position i by l is replaced by a discard at i+k-1 and an expansion by l+1 at i+k-1
#   edge cases that do not require a discard:
#        replaced by an expansion by l at i+k if i+k == 0
#        replaced by an expansion by l at i+k-1 if i+k==L or i+k-1 is already in the discarded positions
#        if all positions are discarded (and the first expansion would add l match states to a model of length 0)
#        the length of the expansion is reduced by 1
#
# k can be any integer
# L is the length of the array to which the indices of pos_expand and pos_discard belong
def extend_mods(pos_expand, expansion_lens, pos_discard, L, k=0):
    if pos_discard.size == L and pos_expand.size > 0:
        expansion_lens = np.copy(expansion_lens)
        expansion_lens[0] -= 1
    if pos_discard.size > 0:
        #find starting points of all consecutive segments of discards
        pos_discard_shift = pos_discard + k
        diff = np.diff(pos_discard_shift, prepend=-1)
        diff_where = np.squeeze(np.argwhere(diff > 1))
        segment_starts = np.atleast_1d(pos_discard_shift[diff_where])
        new_pos_discard = np.insert(pos_discard_shift, diff_where, segment_starts-1)
        new_pos_discard = np.unique(new_pos_discard)
        if pos_discard_shift[-1] == L-1:
            new_pos_discard = new_pos_discard[:-1]
            segment_starts = segment_starts[:-1]
        new_pos_expand = segment_starts-1
        new_expansion_lens = np.ones(segment_starts.size, dtype=expansion_lens.dtype)
    else:
        new_pos_discard = pos_discard
        new_pos_expand = np.array([], dtype=pos_expand.dtype)
        new_expansion_lens = np.array([], dtype=expansion_lens.dtype)
    #handle expansions
    if pos_expand.size > 0:
        pos_expand_shift = pos_expand+k
        extend1 = pos_expand_shift > 0
        extend2 = pos_expand_shift < L
        _,indices,_ = np.intersect1d(pos_expand_shift-1,
                                     np.setdiff1d(np.arange(L), new_pos_discard),
                                     return_indices=True)
        extend3 = np.zeros(pos_expand_shift.size)
        extend3[indices] = 1
        extend = (extend1*extend2*extend3).astype(bool)
        pos_expand_shift[extend1] -= 1
        adj_expansion_lens = np.copy(expansion_lens)
        adj_expansion_lens[extend] += 1
        if new_pos_expand.size == 0:
            new_pos_expand = pos_expand_shift
            new_expansion_lens = adj_expansion_lens
        else:
            if pos_expand_shift.size > 1 and pos_expand_shift[0] == 0 and pos_expand_shift[1] == 0:
                adj_expansion_lens[0] += adj_expansion_lens[1]
            for i in new_pos_expand:
                a = np.argwhere(pos_expand_shift == i)
                if a.size > 0:
                    adj_expansion_lens[a[0]] += 1
            new_pos_expand = np.concatenate([pos_expand_shift, new_pos_expand])
            new_expansion_lens = np.concatenate([adj_expansion_lens, new_expansion_lens])
            new_pos_expand, indices = np.unique(new_pos_expand, return_index=True)
            new_expansion_lens = new_expansion_lens[indices]
        if new_pos_discard.size > 0:
            new_pos_discard = np.concatenate([new_pos_discard,
                                              pos_expand_shift[extend]])
            new_pos_discard = np.unique(new_pos_discard)
        else:
            new_pos_discard = pos_expand_shift[extend]
    return new_pos_expand, new_expansion_lens, new_pos_discard


#applies expansions and discards to emission and transition kernels
def update_kernels(
    am,
    model_index,
    pos_expand,
    expansion_lens,
    pos_discard,
    emission_dummy,
    transition_dummy,
    init_flank_dummy,
):
    L = am.msa_hmm_layer.cell.length[model_index]
    emissions = [em.emission_kernel[model_index].numpy() for em in am.msa_hmm_layer.cell.emitter]
    transitions = { key : kernel.numpy()
                         for key, kernel in am.msa_hmm_layer.cell.transitioner.transition_kernel[model_index].items()}
    dtype = am.msa_hmm_layer.cell.dtype
    emission_dummy = [d((1, em.shape[-1]), dtype).numpy() for d,em in zip(emission_dummy, emissions)]
    transition_dummy = { key : transition_dummy[key](t.shape, dtype).numpy() for key, t in transitions.items()}
    init_flank_dummy = init_flank_dummy((1), dtype).numpy()
    emissions_new = [apply_mods(k,
                                  pos_expand,
                                  expansion_lens,
                                  pos_discard,
                                  d) for k,d in zip(emissions, emission_dummy)]
    transitions_new = {}
    args1 = extend_mods(pos_expand,expansion_lens,pos_discard,L)
    transitions_new["match_to_match"] = apply_mods(transitions["match_to_match"],
                                                      *args1,
                                                      transition_dummy["match_to_match"][0])
    transitions_new["match_to_insert"] = apply_mods(transitions["match_to_insert"],
                                                      *args1,
                                                      transition_dummy["match_to_insert"][0])
    transitions_new["insert_to_match"] = apply_mods(transitions["insert_to_match"],
                                                      *args1,
                                                      transition_dummy["insert_to_match"][0])
    transitions_new["insert_to_insert"] = apply_mods(transitions["insert_to_insert"],
                                                      *args1,
                                                      transition_dummy["insert_to_insert"][0])
    args2 = extend_mods(pos_expand,expansion_lens,pos_discard,L+1,k=1)
    transitions_new["match_to_delete"] = apply_mods(transitions["match_to_delete"],
                                                     *args2,
                                                      transition_dummy["match_to_delete"][0])
    args3 = extend_mods(pos_expand,expansion_lens,pos_discard,L+1)
    transitions_new["delete_to_match"] = apply_mods(transitions["delete_to_match"],
                                                     *args3,
                                                      transition_dummy["delete_to_match"][0])
    transitions_new["delete_to_delete"] = apply_mods(transitions["delete_to_delete"],
                                                     *args1,
                                                      transition_dummy["delete_to_delete"][0])

    #always reset the multi-hit transitions:
    transitions_new["left_flank_loop"] = transition_dummy["left_flank_loop"]
    transitions_new["left_flank_exit"] = transition_dummy["left_flank_exit"]
    init_flank_new = init_flank_dummy
    transitions_new["right_flank_loop"] = transition_dummy["right_flank_loop"]
    transitions_new["right_flank_exit"] = transition_dummy["right_flank_exit"]
    transitions_new["end_to_unannotated_segment"] = transition_dummy["end_to_unannotated_segment"]
    transitions_new["end_to_right_flank"] = transition_dummy["end_to_right_flank"]
    transitions_new["end_to_terminal"] = transition_dummy["end_to_terminal"]
    transitions_new["unannotated_segment_loop"] = transition_dummy["unannotated_segment_loop"]
    transitions_new["unannotated_segment_exit"] = transition_dummy["unannotated_segment_exit"]

    # Maybe TODO?: Discarding or extending positions has the side effect of changing all probabilities
    # in begin-state transition distribution. E.g.
    # Depending on discarded positions, adjust weights such that the residual distribution after
    # discarding some match states is unaffected.
    # If an insert position is expanded, the transitions from begin to the new match states should have
    # probabilities according to the initial dummy distribution and the weights of the old transitions
    # should also be corrected accordingly.

    transitions_new["begin_to_match"] = apply_mods(transitions["begin_to_match"],
                                                      pos_expand,
                                                      expansion_lens,
                                                      pos_discard,
                                                      transition_dummy["begin_to_match"][1])
    if 0 in pos_expand:
        transitions_new["begin_to_match"][0] = transition_dummy["begin_to_match"][0]

    if L in pos_expand:
        transitions["match_to_end"][-1] = transition_dummy["match_to_end"][0]
    transitions_new["match_to_end"] = apply_mods(transitions["match_to_end"],
                                                  pos_expand,
                                                  expansion_lens,
                                                  pos_discard,
                                                  transition_dummy["match_to_end"][0])
    return transitions_new, emissions_new, init_flank_new


@dataclass
class ModelSurgeryResult:
    emitter: list[tf.keras.layers.Layer]
    transitioner: tf.keras.layers.Layer
    model_lengths: np.ndarray
    surgery_converged: bool


def do_model_surgery(
    am: AlignmentModel,
    surgery_del: float,
    surgery_ins: float,
    emission_dummy: list[initializers.Initializer],
    transition_dummy: dict[str, initializers.Initializer],
    flank_init_dummy: initializers.Initializer,
    verbose: bool=False
) -> ModelSurgeryResult:
    surgery_converged = True
    # Duplicate the previous emitters and transitioner and replace their
    # initializers later
    emitter = [em.duplicate() for em in am.msa_hmm_layer.cell.emitter]
    transitioner = am.msa_hmm_layer.cell.transitioner.duplicate()
    pos_expand, expansion_lens, pos_discard = get_discard_or_expand_positions(
        am,
        del_t=surgery_del,
        ins_t=surgery_ins
    )
    model_lengths = []
    for i,k in enumerate(range(am.num_models)):
        surgery_converged &= pos_expand[k].size == 0 and pos_discard[k].size == 0
        if verbose:
            if pos_expand[k].size > 0:
                print(
                    f"expansions model {i}:",
                    list(zip(pos_expand[k], expansion_lens[k]))
                )
            if len(pos_discard[k]) > 0:
                print(
                    f"discards model {i}:", pos_discard[k]
                )
        transition_init, emission_init, flank_init = update_kernels(
            am,
            k,
            pos_expand[k],
            expansion_lens[k],
            pos_discard[k],
            emission_dummy,
            transition_dummy,
            flank_init_dummy,
        )
        for em, old_em, e_init in zip(
            emitter, am.msa_hmm_layer.cell.emitter, emission_init
        ):
            em.emission_init[i] = initializers.ConstantInitializer(e_init)
            em.insertion_init[i] = initializers.ConstantInitializer(
                old_em.insertion_kernel[k].numpy()
            )
        transitioner.transition_init[i] = {
            key : initializers.ConstantInitializer(t)
            for key,t in transition_init.items()
        }
        transitioner.flank_init[i] = initializers.ConstantInitializer(flank_init)
        model_lengths.append(emission_init[0].shape[0])
        if model_lengths[-1] < 3:
            raise SystemExit(
                "A problem occured during model surgery: "\
                "A pHMM is too short (length <= 2)."
            )
    return ModelSurgeryResult(
        emitter=emitter,
        transitioner=transitioner,
        model_lengths=model_lengths,
        surgery_converged=surgery_converged
    )