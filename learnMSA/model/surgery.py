from dataclasses import dataclass

import numpy as np

from learnMSA.config.hmm import PHMMConfig
from learnMSA.config.language_model import LanguageModelConfig
from learnMSA.config.util import get_value
from learnMSA.hmm.tf.layer import PHMMLayer
from learnMSA.hmm.util.transition_index_set import PHMMTransitionIndexSet
from learnMSA.model.tf.model import LearnMSAModel
from learnMSA.util.sequence_dataset import SequenceDataset


def get_discard_or_expand_positions(
    model: LearnMSAModel,
    data: SequenceDataset,
    indices: np.ndarray|None = None,
    del_t: float = 0.5,
    ins_t: float = 0.5,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """ Given an AlignmentModel, computes positions for match expansions and
    discards based on the posterior state probabilities.

    Args:
        model: A LearnMSAModel object with a PHMMLayer.
        data: A SequenceDataset used for computing the posterior state.
        indices: Optional indices to select a subset of the data. If None, all
            sequences in `data` are used.
        del_t: This number is compared to the expected number of times a match
            state is used when aligning a protein from the underlying dataset
            to the pHMM.
        ins_t: This number is compared to the expected number of times an insert
            state is used when aligning a protein from the underlying dataset
            to the pHMM.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            pos_expand: A list of arrays with match positions to expand per model.
            expansion_lens: A list of arrays with the expansion lengths.
            pos_discard: A list of arrays with match positions to discard.
    """
    # num_models x max_num_states
    model.posterior_mode()
    model.compile()
    expected_state = model.predict(data, indices, reduce=True) # (H, Q)
    pos_expand = []
    expansion_lens = []
    pos_discard = []
    for i in range(model.heads):
        model_length = model.lengths[i]
        #discards
        match_states = expected_state[i, :model_length]
        discard = np.arange(model_length, dtype=np.int32)[match_states < del_t]
        pos_discard.append(discard)
        #expansions
        insert_states = expected_state[i, model_length:2*model_length-1]
        left_flank_state = expected_state[i, 2*model_length-1]
        right_flank_state = expected_state[i, 2*model_length+1]
        all_inserts = np.concatenate(
            [[left_flank_state], insert_states, [right_flank_state]], axis=0
        )
        which_to_expand = all_inserts > ins_t
        expand = np.arange(model_length+1, dtype=np.int32)[which_to_expand]
        pos_expand.append(expand)
        #expansion lengths
        expand_len = np.ceil(all_inserts).astype(np.int32)[which_to_expand]
        expansion_lens.append(expand_len)
    return pos_expand, expansion_lens, pos_discard


def apply_mods(
    x: np.ndarray,
    pos_expand: np.ndarray,
    expansion_lens: np.ndarray,
    pos_discard: np.ndarray,
    insert_value: np.ndarray|float|int,
    del_marker: float = -9999.0,
):
    """
    Applies modifications (discards and expansions) to axis 0 of x.

    Args:
        x (np.ndarray): The input array to modify of shape `(N, ...)`.
        pos_expand (np.ndarray): Positions to expand of shape ``(K,)``, where
            `K <= N` is the number of positions to expand.
        expansion_lens (np.ndarray): Lengths of expansions of shape ``(K,)``
            corresponding to `pos_expand`.
        pos_discard (np.ndarray): Positions to discard of shape ``(M,)``, where
            `M <= N` is the number of positions to discard.
        insert_value (np.ndarray): Value to insert for expansions
            of shape ``(...,)``.
        del_marker (float, optional): Marker value for discards.
            Defaults to -9999.0. Should not occur in x otherwise.

    Returns:
        (np.ndarray): A copy of x with modifications applied.
    """
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


def extend_mods(
    pos_expand: np.ndarray,
    expansion_lens: np.ndarray,
    pos_discard: np.ndarray,
    L: int,
    k: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function transforms modification (discards and expansions) vectors to
    fulfill specific alignment rules:

    - Each consecutive segment of discards from i to j is replaced with discards
      from i+k-1 to j+k and an expansion of length 1 at i+k-1.
      Edge cases that do not require an expansion:
        * Replaced with discards from i+k to j+k if i+k == 0 and j+k < L-1
        * Replaced with discards from i+k-1 to j+k-1 if i+k > 0 and j+k == L-1
        * Replaced with discards from i+k to j+k-1 if i+k == 0 and j+k == L-1

    - An expansion at position i by l is replaced by a discard at i+k-1 and an
      expansion by l+1 at i+k-1.
      Edge cases that do not require a discard:
        * Replaced by an expansion by l at i+k if i+k == 0
        * Replaced by an expansion by l at i+k-1 if i+k==L or i+k-1 is already
          in the discarded positions
        * If all positions are discarded (and the first expansion would add l
          match states to a model of length 0), the length of the expansion is
          reduced by 1

    Args:
        pos_expand (np.ndarray): Positions to expand of shape ``(K,)``.
        expansion_lens (np.ndarray): Lengths of expansions of shape ``(K,)``
            corresponding to `pos_expand`.
        pos_discard (np.ndarray): Positions to discard of shape ``(M,)``.
        L (int): The length of the array to which the indices of pos_expand
            and pos_discard belong.
        k (int, optional): Offset to shift positions. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - new_pos_expand: Updated positions to expand
            - new_expansion_lens: Updated expansion lengths
            - new_pos_discard: Updated positions to discard
    """
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

@dataclass
class UpdateKernelResult:
    length: int
    """The lengths of the updated models."""
    config: PHMMConfig
    """The updated PHMMConfig with modified parameters."""
    plm_config: LanguageModelConfig | None = None
    """The updated LanguageModelConfig with modified parameters, if
    the PHMMLayer was using embeddings."""

def update_kernels(
    phmm_layer: PHMMLayer,
    model_index: int,
    pos_expand: np.ndarray,
    expansion_lens: np.ndarray,
    pos_discard: np.ndarray,
    config: PHMMConfig,
    plm_config: LanguageModelConfig | None = None,
) -> UpdateKernelResult:
    """
    Apply expansions and discards to emission and transition kernels.

    This function modifies the model kernels according to specified position
    expansions and discards, using values from the model configuration for
    newly created positions (i.e. newly added states are initialized in the
    same way they would have been initialized in the original model).

    Note:
        The function handles different transition types with appropriate
        position shifts using `extend_mods`. Transitions from the flanks and
        unannotated segment states are always reset to initial values.

    Args:
        phmm_layer (PHMMLayer): The layer containing the kernels to update.
        model_index (int): The index of the model to update.
        pos_expand (np.ndarray): Positions to expand of shape ``(K,)``, where
            `K`must be less than or equal to the number of match states.
        expansion_lens (np.ndarray): Lengths of expansions of shape ``(K,)``
            corresponding to `pos_expand`.
        pos_discard (np.ndarray): Positions to discard of shape ``(M,)``,
            where `M` must be less than the number of match states..
        config (PHMMConfig): The configuration containing the initialization
            values.
        plm_config (LanguageModelConfig | None): The protein language model
            configuration.

    Returns:
        PHMMConfig: A new configuration with modified parameters that can be
            used to create a new PHMMLayer with updated kernels.
    """
    head_subset_backup = phmm_layer.head_subset
    phmm_layer.head_subset = [model_index]
    L = phmm_layer.lengths[model_index]

    # Gather the current emission parameters
    aa_emissions = phmm_layer.hmm.emitter[0].matrix().numpy() # (1, Q, S)
    if phmm_layer.use_language_model:
        emb_emissions = phmm_layer.hmm.emitter[1].matrix().numpy()  # (1, Q, 2D)

    # Slice to the model of interest and to only match states
    aa_emissions = aa_emissions[0, :L, :]  # (L, S)
    if phmm_layer.use_language_model:
        emb_emissions = emb_emissions[0, :L, :]  # (L, 2D)

    # Apply modifications to the amino acid emission parameters
    aa_insert_value = np.array(config.background_distribution)
    aa_emissions_new = apply_mods(
        aa_emissions,
        pos_expand=pos_expand,
        expansion_lens=expansion_lens,
        pos_discard=pos_discard,
        insert_value=aa_insert_value,
    )

    # Apply modifications to the embedding emission parameters
    if phmm_layer.use_language_model:
        assert plm_config is not None,\
            "plm_config must be provided to update_kernels if"\
            "the PHMMLayer uses a language model."
        embedding_dim = plm_config.scoring_model_dim
        emb_expectations = np.zeros((embedding_dim,), dtype=np.float32)
        # TODO: This should ideally use different random variances per
        # new position
        emb_stddev = np.random.normal(
            0.0, plm_config.variance_init_stdev, (embedding_dim,)
        ).astype(np.float32)
        emb_insert_value = np.concatenate(
            [emb_expectations, emb_stddev], axis=0
        )
        emb_emissions_new = apply_mods(
            emb_emissions,
            pos_expand=pos_expand,
            expansion_lens=expansion_lens,
            pos_discard=pos_discard,
            insert_value=emb_insert_value,
        )

    # Gather the current transition parameters
    # Note: The transitions not occuring here (like left_flank_loop) are
    # always reset to initial values later on.
    A = phmm_layer.hmm.transitioner.explicit_transitioner.matrix().numpy()
    A = A[0] # (Q, Q)
    ind = PHMMTransitionIndexSet(L)
    match_to_match = A[ind.match_to_match[:, 0], ind.match_to_match[:, 1]]
    match_to_insert = A[ind.match_to_insert[:, 0], ind.match_to_insert[:, 1]]
    insert_to_insert = A[ind.insert_to_insert[:, 0], ind.insert_to_insert[:, 1]]
    delete_to_delete = A[ind.delete_to_delete[:, 0], ind.delete_to_delete[:, 1]]
    begin_to_match = A[ind.begin_to_match[:, 0], ind.begin_to_match[:, 1]]
    match_to_end = A[ind.match_to_end[:, 0], ind.match_to_end[:, 1]]
    begin_to_delete = A[ind.begin_to_delete[:, 0], ind.begin_to_delete[:, 1]]

    h = model_index
    args = extend_mods(pos_expand, expansion_lens, pos_discard, L)

    match_to_match = apply_mods(
        match_to_match, *args,
        insert_value = get_value(config.p_match_match, h, 0),
    )
    match_to_insert = apply_mods(
        match_to_insert, *args,
        insert_value = get_value(config.p_match_insert, h, 0),
    )
    insert_to_insert = apply_mods(
        insert_to_insert, *args,
        insert_value = get_value(config.p_insert_insert, h, 0),
    )
    delete_to_delete = apply_mods(
        delete_to_delete, *args,
        insert_value = get_value(config.p_delete_delete, h, 0),
    )

    begin_to_match = apply_mods(
        begin_to_match,
        pos_expand,
        expansion_lens,
        pos_discard,
        get_value(config.p_begin_match, h, 1),
    )
    if 0 in pos_expand:
        begin_to_match[0] = get_value(config.p_begin_match, h, 0)
        begin_to_delete = get_value(config.p_begin_delete, h)
    # Re-normalize the internal begin_to_match probabilities
    p1 = begin_to_match[1:].sum()
    p2 = 1 - begin_to_match[0] - begin_to_delete
    if p1 > 0 and p2 > 0:
        begin_to_match[1:] /= p1 / p2
    elif p1 > 0 and p2 <= 0:
        # If p2 <= 0, the first position and delete consume all probability
        # Set internal positions to very small values
        begin_to_match[1:] = 1e-10
    # else: p1 == 0, keep begin_to_match[1:] as is (should be all zeros)

    if config.p_match_end is None:
        p_match_end = 0.5 / (L - 1) if L > 2 else 0.0
    else:
        p_match_end = get_value(config.p_match_end, h, 0)

    if L in pos_expand:
        match_to_end[-1] = p_match_end
    match_to_end = apply_mods(
        match_to_end,
        pos_expand,
        expansion_lens,
        pos_discard,
        p_match_end,
    )

    new_config = config.model_copy(deep=True)
    new_config.match_emissions=aa_emissions_new[np.newaxis]
    new_config.p_match_match=match_to_match[np.newaxis]
    new_config.p_match_insert=match_to_insert[np.newaxis]
    new_config.p_insert_insert=insert_to_insert[np.newaxis]
    new_config.p_delete_delete=delete_to_delete[np.newaxis]
    new_config.p_begin_match=begin_to_match[np.newaxis]
    new_config.p_match_end=match_to_end[np.newaxis]
    new_config.p_begin_delete=begin_to_delete

    if phmm_layer.use_language_model and plm_config is not None:
        new_plm_config = plm_config.model_copy(deep=True)
        new_plm_config.match_expectations = emb_emissions_new[:, :embedding_dim]
        new_plm_config.match_stddev = emb_emissions_new[:, embedding_dim:]
    else:
        new_plm_config = None

    # Reset
    phmm_layer.head_subset = head_subset_backup

    return UpdateKernelResult(
        length=aa_emissions_new.shape[0],
        config=new_config,
        plm_config=new_plm_config
    )

@dataclass
class ModelSurgeryResult:
    model_lengths: np.ndarray
    """The lengths of the updated models."""
    surgery_converged: bool
    """Whether no modifications were applied during surgery."""
    config: PHMMConfig
    """The updated PHMMConfig with modified parameters."""
    plm_config: LanguageModelConfig | None = None
    """The updated LanguageModelConfig with modified parameters, if
    the PHMMLayer was using embeddings."""

def model_surgery(
    model: LearnMSAModel,
    data: SequenceDataset,
    indices: np.ndarray|None = None,
    surgery_del: float = 0.5,
    surgery_ins: float = 0.5,
    verbose: bool = False,
) -> ModelSurgeryResult:
    """
    A heuristic that optimizes the length of the pHMMs in the given PHMMLayer
    based on the posterior probabilities of states.

    Args:
        model: A LearnMSAModel object with a PHMMLayer.
        data: A SequenceDataset used for computing the posterior state.
        indices: Optional indices to select a subset of the data. If None, all
            sequences in `data` are used.
        surgery_del: Discards match states for which `surgery_del` is larger
            than the expected number of times this match state is used in an
            alignment of a sequence from the underlying dataset to the model.
        surgery_ins: Discards match states for which `surgery_ins` is smaller
            than the expected number of times insert states between two match
            states are used in an alignment of a sequence from the underlying
            dataset to the model. New match states are added according to the
            expected insertion length.
        verbose: Whether to print information about the surgery process.

    Returns:
        A ModelSurgeryResult object.
    """

    # Find positions to discard or expand
    pos_expand, expansion_lens, pos_discard = get_discard_or_expand_positions(
        model=model,
        data=data,
        indices=indices,
        del_t=surgery_del,
        ins_t=surgery_ins
    )

    # Loop over models and apply modifications
    surgery_converged = True #becomes False if any modification is applied
    model_lengths = []
    configs = []
    plm_configs = []
    for i,k in enumerate(range(model.heads)):
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

        result = update_kernels(
            model.phmm_layer,
            k,
            pos_expand[k],
            expansion_lens[k],
            pos_discard[k],
            model.context.config.hmm,
            model.context.config.language_model,
        )

        model_lengths.append(result.length)
        if model_lengths[-1] < 3:
            raise SystemExit(
                "A problem occured during model surgery: "\
                "A pHMM is too short (length <= 2)."
            )

        configs.append(result.config)
        plm_configs.append(result.plm_config)

    # Merge configurations that contain parameters per head to a single config
    def concat_param(param_name: str):
        values = [getattr(c, param_name) for c in configs]

        # Check if all values are None
        if all(v is None for v in values):
            return None

        arrays = [np.atleast_1d(v) for v in values]

        # Simple 1D case (like p_begin_delete): just concatenate
        if arrays[0].ndim == 1:
            return np.concatenate(arrays, axis=0)

        # 2D or higher: create zeros array and fill
        num_heads = len(arrays)
        max_len = max(arr.shape[1] for arr in arrays)
        full_shape = (num_heads, max_len) + arrays[0].shape[2:]
        result = np.zeros(full_shape, dtype=arrays[0].dtype)

        for i, arr in enumerate(arrays):
            result[i, :arr.shape[1]] = arr[0]

        return result

    merged_config = configs[0].model_copy(deep=True)
    merged_config.match_emissions = concat_param("match_emissions")
    merged_config.insert_emissions = concat_param("insert_emissions")
    merged_config.p_begin_match = concat_param("p_begin_match")
    merged_config.p_match_match = concat_param("p_match_match")
    merged_config.p_match_insert = concat_param("p_match_insert")
    merged_config.p_match_end = concat_param("p_match_end")
    merged_config.p_insert_insert = concat_param("p_insert_insert")
    merged_config.p_delete_delete = concat_param("p_delete_delete")
    merged_config.p_begin_delete = concat_param("p_begin_delete")
    merged_config.p_left_left = concat_param("p_left_left")
    merged_config.p_right_right = concat_param("p_right_right")
    merged_config.p_unannot_unannot = concat_param("p_unannot_unannot")
    merged_config.p_end_unannot = concat_param("p_end_unannot")
    merged_config.p_end_right = concat_param("p_end_right")
    merged_config.p_start_left_flank = concat_param("p_start_left_flank")

    if plm_configs[0] is not None:
        merged_plm_config = plm_configs[0].model_copy(deep=True)
        merged_plm_config.match_expectations = np.concatenate(
            [c.match_expectations for c in plm_configs], axis=0
        )
        merged_plm_config.match_stddev = np.concatenate(
            [c.match_stddev for c in plm_configs], axis=0
        )
        merged_plm_config.insert_expectations = np.concatenate(
            [c.insert_expectations for c in plm_configs], axis=0
        )
        merged_plm_config.insert_stddev = np.concatenate(
            [c.insert_stddev for c in plm_configs], axis=0
        )
    else:
        merged_plm_config = None

    return ModelSurgeryResult(
        model_lengths=np.array(model_lengths, dtype=np.int32),
        surgery_converged=surgery_converged,
        config=merged_config,
        plm_config=merged_plm_config,
    )