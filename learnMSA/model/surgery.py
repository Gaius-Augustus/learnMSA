from dataclasses import dataclass
from typing import Sequence

import numpy as np

from learnMSA.config.hmm import PHMMConfig
from learnMSA.config.language_model import LanguageModelConfig
from learnMSA.config.structure import StructureConfig
from learnMSA.config.training import TrainingConfig
from learnMSA.config.util import get_value
from learnMSA.hmm.tf.joint_profile_emitter import outer_product_flat_pw
from learnMSA.hmm.tf.layer import PHMMLayer
from learnMSA.hmm.tf.util import load_dirichlet
from learnMSA.hmm.util.transition_index_set import PHMMTransitionIndexSet
from learnMSA.hmm.util.value_set import PHMMValueSet
from learnMSA.hmm.util.value_set_emb import PHMMEmbeddingValueSet
from learnMSA.model.tf.model import LearnMSAModel
from learnMSA.util.dataset import Dataset
from learnMSA.util.sequence_dataset import SequenceDataset


def get_discard_or_expand_positions(
    model: LearnMSAModel,
    data: SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
    indices: np.ndarray|None = None,
    del_t: float = 0.5,
    ins_t: float = 0.5,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """ Given an AlignmentModel, computes positions for match expansions and
    discards based on the posterior state probabilities.

    Args:
        model: A LearnMSAModel object with a PHMMLayer.
        data: A SequenceDataset or tuple of Dataset(s) used for computing the
            posterior state.
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
    expected_state = model.predict(data, indices=indices, reduce=True) # (H, Q)
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
    aa_values: PHMMValueSet | None = None
    """The updated pHMM parameters."""
    emb_values: PHMMEmbeddingValueSet | None = None
    """The updated embedding value sets."""
    struct_values: PHMMValueSet | None = None
    """The updated structural value sets."""
    joint_aa_struct_values: PHMMValueSet | None = None
    """Can be provided instead of aa_values and struct_values if the model uses
    joint emissions for amino acids and structural information."""

def update_kernels(
    phmm_layer: PHMMLayer,
    model_index: int,
    pos_expand: np.ndarray,
    expansion_lens: np.ndarray,
    pos_discard: np.ndarray,
    config: PHMMConfig,
    training_config: TrainingConfig,
    plm_config: LanguageModelConfig | None = None,
    structural_config: StructureConfig | None = None,
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
        training_config (TrainingConfig): The training configuration containing
            parameters that control the behavior of the model during training.
        plm_config (LanguageModelConfig | None): The protein language model
            configuration.
        structural_config (StructureConfig | None): The structural information
            configuration.

    Returns:
        PHMMConfig: A new configuration with modified parameters that can be
            used to create a new PHMMLayer with updated kernels.
    """
    head_subset_backup = phmm_layer.head_subset
    phmm_layer.head_subset = [model_index]
    L = phmm_layer.lengths[model_index]

    # Amino acids
    if not phmm_layer.no_aa\
            and not training_config.reset_emissions_after_surgery\
            and phmm_layer.joint_emitter is None:
        assert phmm_layer.profile_emitter is not None
        aa_emissions = phmm_layer.profile_emitter.matrix().numpy()
        assert aa_emissions.shape[0] == 1,\
            "Head subset is not working properly for the amino acid emitter."
        aa_emissions = aa_emissions[0, :L, :]
        aa_insert_value = _get_aa_insert_value(config)
        aa_emissions_new = apply_mods(
            aa_emissions,
            pos_expand=pos_expand,
            expansion_lens=expansion_lens,
            pos_discard=pos_discard,
            insert_value=aa_insert_value,
        )
    else:
        aa_emissions_new = None

    # Structural information
    if phmm_layer.use_structure\
            and structural_config is not None\
            and not structural_config.reset_after_surgery\
            and phmm_layer.joint_emitter is None:
        assert phmm_layer.struct_emitter is not None
        struct_emissions = phmm_layer.struct_emitter.matrix().numpy()
        assert struct_emissions.shape[0] == 1,\
            "Head subset is not working properly for the structural emitter."
        struct_emissions = struct_emissions[0, :L, :]
        struct_insert_value = _get_struct_insert_value(structural_config)
        struct_emissions_new = apply_mods(
            struct_emissions,
            pos_expand=pos_expand,
            expansion_lens=expansion_lens,
            pos_discard=pos_discard,
            insert_value=struct_insert_value,
        )
    else:
        struct_emissions_new = None

    # pLM embeddings
    if phmm_layer.use_language_model:
        assert phmm_layer.embedding_emitter is not None
        emb_emissions = phmm_layer.embedding_emitter.matrix().numpy()
        assert emb_emissions.shape[0] == 1,\
            "Head subset is not working properly for the embedding emitter."
        emb_emissions = emb_emissions[0, :L, :]
        assert plm_config is not None,\
            "plm_config must be provided to update_kernels if"\
            "the PHMMLayer uses a language model."
        embedding_dim = plm_config.scoring_model_dim
        if hasattr(phmm_layer, "emb_mean"):
            emb_expectations = phmm_layer.emb_mean
        else:
            emb_expectations = np.zeros((embedding_dim,), dtype=np.float32)
        emb_var = np.zeros((embedding_dim,), dtype=np.float32)
        emb_var += plm_config.variance_init
        emb_insert_value = np.concatenate([emb_expectations, emb_var], axis=0) # type: ignore
        emb_emissions_new = apply_mods(
            emb_emissions,
            pos_expand=pos_expand,
            expansion_lens=expansion_lens,
            pos_discard=pos_discard,
            insert_value=emb_insert_value,
        )
    else:
        emb_emissions_new = None

    # Joint emissions
    if phmm_layer.joint_emitter is not None:
        joint_emissions = phmm_layer.joint_emitter.matrix().numpy()
        assert joint_emissions.shape[0] == 1,\
            "Head subset is not working properly for the joint emitter."
        joint_emissions = joint_emissions[0, :L, :]

        aa_insert_value = _get_aa_insert_value(config)
        assert structural_config is not None,\
            "structural_config must be provided to update_kernels if the "\
            "PHMMLayer uses joint emissions."
        struct_insert_value = _get_struct_insert_value(structural_config)

        joint_insert_value = outer_product_flat_pw(
            aa_insert_value, struct_insert_value
        ).numpy()
        joint_emissions_new = apply_mods(
            joint_emissions,
            pos_expand=pos_expand,
            expansion_lens=expansion_lens,
            pos_discard=pos_discard,
            insert_value=joint_insert_value,
        )

    L_new = L - pos_discard.size + int(expansion_lens.sum())

    if not training_config.reset_transitions_after_surgery:
        # Gather the current transition parameters
        # Note: The transitions not occuring here (like left_flank_loop) are
        # always reset to initial values later on.
        # Note 2: explicit_transitioner does not have the head subset!
        A = phmm_layer.hmm.transitioner.explicit_transitioner.matrix().numpy()
        A = A[model_index] # (Q, Q)
        ind = PHMMTransitionIndexSet(L)
        match_to_match = A[ind.match_to_match[:, 0], ind.match_to_match[:, 1]]
        match_to_insert = A[ind.match_to_insert[:, 0], ind.match_to_insert[:, 1]]
        match_to_delete = A[ind.match_to_delete[:, 0], ind.match_to_delete[:, 1]]
        insert_to_insert = A[ind.insert_to_insert[:, 0], ind.insert_to_insert[:, 1]]
        delete_to_delete = A[ind.delete_to_delete[:, 0], ind.delete_to_delete[:, 1]]

        # Preserve Begin -> {M1, D1}, but reset entry probabilities uniformly
        # according to the remaining probability mass
        begin_match = A[ind.begin_to_match[:, 0], ind.begin_to_match[:, 1]]
        begin_delete = A[ind.begin_to_delete[0,0], ind.begin_to_delete[0,1]]
        match_end = A[ind.match_to_end[:, 0], ind.match_to_end[:, 1]]

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
        match_to_delete = apply_mods(
            match_to_delete, *args,
            insert_value = get_value(config.p_match_delete, h, 0),
        )

        if 0 in pos_discard or 0 in pos_expand:
            # Reset if we modified the first match state
            begin_match_1 = get_value(config.p_begin_match, h, 0)
            begin_delete = get_value(config.p_begin_delete, h, 0)
        else:
            begin_match_1 = begin_match[0]
        if L_new > 1:
            begin_to_match_insert = (1 - begin_match_1 - begin_delete) / (L_new-1)
        else:
            begin_to_match_insert = 0.0

        begin_match = apply_mods(
            begin_match, pos_expand, expansion_lens, pos_discard,
            insert_value = begin_to_match_insert,
        )
        begin_match[0] = begin_match_1

        if L in pos_expand:
            # avoid moving p last_match -> end = 1.0 into the model
            match_end[-1] = begin_to_match_insert
        match_end = apply_mods(
            match_end, pos_expand, expansion_lens, pos_discard,
            insert_value = begin_to_match_insert,
        )

        # re-normalize begin probabilities
        begin_match[1:] /= begin_match[1:].sum() / (1 - begin_match[0] - begin_delete)

    new_config = config.model_copy(deep=True)
    if aa_emissions_new is not None:
        new_config.match_emissions=aa_emissions_new[np.newaxis]
    if not training_config.reset_transitions_after_surgery:
        new_config.p_match_match=match_to_match[np.newaxis]
        new_config.p_match_insert=match_to_insert[np.newaxis]
        new_config.p_match_delete=match_to_delete[np.newaxis]
        new_config.p_insert_insert=insert_to_insert[np.newaxis]
        new_config.p_delete_delete=delete_to_delete[np.newaxis]
        new_config.p_begin_match = begin_match if isinstance(begin_match, float) else begin_match[np.newaxis]
        new_config.p_begin_delete = begin_delete
        new_config.p_match_end = match_end[np.newaxis]

    aa_values = PHMMValueSet.from_config(L_new, 0, new_config)

    if phmm_layer.use_language_model\
            and plm_config is not None:
        assert emb_emissions_new is not None
        # TODO: get rid of config detour
        new_plm_config = plm_config.model_copy(deep=True)
        new_plm_config.match_expectations = emb_emissions_new[np.newaxis, :, :embedding_dim]
        new_plm_config.match_variance = emb_emissions_new[np.newaxis, :, embedding_dim:]
        emb_value_sets = PHMMEmbeddingValueSet.from_config(
            L_new, 0, new_plm_config
        )
    else:
        emb_value_sets = None

    if phmm_layer.use_structure\
            and structural_config is not None\
            and not structural_config.reset_after_surgery\
            and not phmm_layer.joint_emitter:
        assert struct_emissions_new is not None
        # TODO: get rid of config detour
        new_structural_config = structural_config.model_copy(deep=True)
        new_structural_config.match_emissions = struct_emissions_new[np.newaxis]
        struct_value_sets = PHMMValueSet.from_structural_config(
            L_new, 0, new_structural_config
        )
    else:
        struct_value_sets = None

    if phmm_layer.joint_emitter is None:
        joint_aa_struct_value_sets = None
    else:
        assert joint_emissions_new is not None
        joint_aa_struct_value_sets = PHMMValueSet(
            L_new, joint_emissions_new, joint_insert_value
        )

    # Reset
    phmm_layer.head_subset = head_subset_backup

    return UpdateKernelResult(
        length=L_new,
        aa_values=aa_values,
        emb_values=emb_value_sets,
        struct_values=struct_value_sets,
        joint_aa_struct_values=joint_aa_struct_value_sets,
    )

@dataclass
class ModelSurgeryResult:
    model_lengths: np.ndarray
    """The lengths of the updated models."""
    surgery_converged: bool
    """Whether no modifications were applied during surgery."""
    aa_values: Sequence[PHMMValueSet] | None
    """The updated pHMM parameters."""
    emb_values: Sequence[PHMMEmbeddingValueSet] | None
    """The updated embedding value sets."""
    struct_values: Sequence[PHMMValueSet] | None
    """The updated structural value sets."""
    joint_aa_struct_values: Sequence[PHMMValueSet] | None
    """Can be provided instead of aa_values and struct_values if the model uses
    joint emissions for amino acids and structural information."""

def model_surgery(
    model: LearnMSAModel,
    data: SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
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
        data: A SequenceDataset or tuple of Dataset(s) used for computing the
            posterior state.
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
    aa_values = []
    emb_values = []
    struct_values = []
    joint_aa_struct_values = []
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
            model.context.config.training,
            model.context.config.language_model,
            model.context.config.structure,
        )

        model_lengths.append(result.length)
        if model_lengths[-1] < 3:
            raise SystemExit(
                "A problem occured during model surgery: "\
                "A pHMM is too short (length <= 2)."
            )

        aa_values.append(result.aa_values)
        emb_values.append(result.emb_values)
        struct_values.append(result.struct_values)
        joint_aa_struct_values.append(result.joint_aa_struct_values)

    aa_values = _squeeze_none(aa_values)
    emb_values = _squeeze_none(emb_values)
    struct_values = _squeeze_none(struct_values)
    joint_aa_struct_values = _squeeze_none(joint_aa_struct_values)

    assert aa_values is not None or emb_values is not None\
        or struct_values is not None or joint_aa_struct_values is not None

    return ModelSurgeryResult(
        model_lengths=np.array(model_lengths, dtype=np.int32),
        surgery_converged=surgery_converged,
        aa_values=aa_values,
        emb_values=emb_values,
        struct_values=struct_values,
        joint_aa_struct_values=joint_aa_struct_values,
    )

def _get_aa_insert_value(config: PHMMConfig) -> np.ndarray:
    if config.use_prior_for_emission_init:
        # use prior mean as brackground distribution
        emission_prior = load_dirichlet(
            f"amino_acid_dirichlet_1.weights", # TODO: use component count from config here
            dim = len(SequenceDataset._default_alphabet)-1,
            states = [1],
        )
        aa_insert_value = emission_prior.mean()[0,0].numpy()
    else:
        aa_insert_value = np.array(config.background_distribution)
    return aa_insert_value

def _get_struct_insert_value(structural_config: StructureConfig) -> np.ndarray:
    if structural_config.use_prior_for_emission_init\
            and structural_config.prior_name:
        struct_prior = load_dirichlet(
            structural_config.prior_name+".weights",
            dim=structural_config.alphabet_size,
            components=structural_config.prior_components,
            states=[1],
        )
        struct_insert_value = struct_prior.mean()[0,0].numpy()
    else:
        struct_insert_value = np.array(
            structural_config.background_distribution # type: ignore
        )
    return struct_insert_value

def _squeeze_none(seq: Sequence | None) -> Sequence | None:
    """Returns None if all elements of seq are None, otherwise returns seq."""
    if seq is None:
        return None
    if all(x is None for x in seq):
        return None
    return seq