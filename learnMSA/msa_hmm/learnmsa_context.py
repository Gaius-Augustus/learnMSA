import os
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tensorflow as tf

import learnMSA.msa_hmm.Emitter as emit
import learnMSA.msa_hmm.Initializers as initializers
import learnMSA.msa_hmm.training as train
import learnMSA.msa_hmm.training_util as training_util
import learnMSA.msa_hmm.Transitioner as trans
import learnMSA.protein_language_models.Common as Common
import learnMSA.protein_language_models.EmbeddingBatchGenerator as EmbeddingBatchGenerator
from learnMSA import Configuration
from learnMSA.msa_hmm import clustering
from learnMSA.protein_language_models.MvnEmitter import (
    AminoAcidPlusMvnEmissionInitializer, MvnEmitter)
from learnMSA.run.util import get_gpu_memory, validate_filepath

from ..msa_hmm.AncProbsLayer import inverse_softplus
from . import MSA2HMM, Priors
from .Initializers import (ConstantInitializer, PHMMInitializerSet,
                           make_initializers_from)
from .SequenceDataset import AlignedDataset, SequenceDataset

# Type alias for model length callback
ModelLengthsCallback = Callable[[SequenceDataset], np.ndarray]
BatchSizeCallback = Callable[[SequenceDataset], int]


class LearnMSAContext:
    """
    Sets up data-dependent context for learning a profile HMM.

    Either a SequenceDataset must be provided or the number of sequences along
    with a configuration that includes initial lengths must be specified.
    A context that is created with a SequenceDataset will be independent of
    the dataset after initialization. In particular, the data is not a
    dependency when serializing the context.

    Attributes:
        data: SequenceDataset containing the sequences to align.
        config: Configuration object with all settings.
        model_lenghts_cb: Callable that takes a SequenceDataset and Configuration
            and returns an array of initial model lengths.
    """

    config: Configuration
    num_seq: int
    model_lengths_cb: ModelLengthsCallback
    model_lengths: np.ndarray
    scoring_model_config: Common.ScoringModelConfig #legacy, will be removed in future
    initializers: PHMMInitializerSet
    emitter: list[tf.keras.Layer]
    transitioner: tf.keras.Layer
    batch_size: int | Callable[[SequenceDataset], int]
    batch_gen: train.BatchGenerator
    sequence_weights: np.ndarray | None
    clusters: Any
    subset: np.ndarray

    """
    Is created from a Configuration and a SequenceDataset to hold all relevant
    context for training a profile model and decoding an alignment.
    """
    def __init__(
        self,
        config: Configuration,
        data: SequenceDataset | None = None,
        num_seq: int | None = None,
        sequence_weights: np.ndarray | None = None,
        clusters: Any = None,
    ) -> None:
        """
        Args:
            config: Immutable configuration object with all settings.
            dataset: SequenceDataset containing the sequences to align. Must be
                provided when the remaining parameters are not provided.
            num_seq: Number of sequences for which this context is created.
                Must only be provided when data is None.
            sequence_weights: Array of sequence weights that can optionally be
                provided when data is None. If not provided, sequence weights
                will be 1.
            clusters: Cluster array with the same length as num_seq. Can be
                provided when data is None.
        """
        self.config = config

        if data is None:
            assert num_seq is not None, (
                "When no data is provided, num_seq must be specified."
            )
            assert config.training.length_init is not None, (
                "When no data is provided, length_init must be specified in "\
                "the configuration."
            )
            self.num_seq = num_seq
        else:
            if num_seq is not None:
                warnings.warn(
                    "num_seq is provided but data is not None. "\
                    "It will be ignored."
                )
            self.num_seq = data.num_seq

        # Needed for legacy reasons, may cleanup in the future
        self.scoring_model_config = self._get_scoring_model_config()

        model_len_cb = None

        # Set up initializers
        if self.config.init_msa.from_msa is None:
            self.initializers = self._setup_initializers()
        else:
            self.initializers, model_len_cb = self._setup_init_msa()
        self._setup_visualization()

        # When not included in the initializers, set up lengths from the config
        if model_len_cb is None:
            model_len_cb = self._setup_lengths()

        # If still not set, use default length callback
        if model_len_cb is None:
            def _default_length_callback(data):
                if self.config.training.max_iterations > 1:
                    len_mul = self.config.training.len_mul
                else:
                    len_mul = 1.0
                return training_util.get_initial_model_lengths(
                    data.seq_lens,
                    self.config.training.length_init_quantile,
                    len_mul,
                    self.config.training.num_model,
                )
            model_len_cb = _default_length_callback

        self.model_lengths_cb = model_len_cb

        # Initialize the model lengths
        if data is not None:
            self.model_lengths = self.model_lengths_cb(data)
        else:
            self.model_lengths = np.array(
                self.config.training.length_init, dtype=np.int32
            )

        if data is not None:
            if self.config.training.auto_crop:
                # Setup cropping length if auto_crop is enabled based on data
                # Has to be done before the batch size setup
                self.config.training.crop = int(np.ceil(
                    self.config.training.auto_crop_scale * np.mean(data.seq_lens)
                ))
        assert isinstance(self.config.training.crop, int)

        self.batch_size = self._setup_batch_size_cb()

        if self.config.language_model.use_language_model:
            self._setup_language_model_specific_settings()

        self.emitter, self.transitioner = self._setup_hmm_components()

        # Set up encoder initialization
        self.encoder_initializer = initializers.make_default_anc_probs_init(
            self.config.training.num_model
        )
        self.encoder_weight_extractor = None
        self.encoder_initializer[0] = ConstantInitializer(
            inverse_softplus(
                np.array(self.config.advanced.initial_distance) + 1e-8
            ).numpy()
        )

        # Set custom transitioner parameters if specified
        transitioners = self.transitioner if hasattr(
            self.transitioner, '__iter__') else [self.transitioner]
        for trans in transitioners:
            trans.prior.alpha_flank = self.config.advanced.alpha_flank
            trans.prior.alpha_single = self.config.advanced.alpha_single
            trans.prior.alpha_global = self.config.advanced.alpha_global
            trans.prior.alpha_flank_compl = self.config.advanced.alpha_flank_compl
            trans.prior.alpha_single_compl = self.config.advanced.alpha_single_compl
            trans.prior.alpha_global_compl = self.config.advanced.alpha_global_compl

        # Adjust training settings automatically if skip_training is set
        if self.config.training.skip_training:
            self.config.training.max_iterations = 1
            self.config.training.epochs = [0]*3

        self.batch_gen = self._get_batch_generator()
        if data is not None:
            self.sequence_weights, self.clusters = self._get_clustering(data)
        else:
            if sequence_weights is None:
                sequence_weights = np.ones((self.num_seq,), dtype=np.float32)
            else:
                assert len(sequence_weights) == self.num_seq, (
                    "Length of sequence_weights does not match num_seq."
                )
            if clusters is not None:
                assert len(clusters) == self.num_seq, (
                    "Length of clusters does not match num_seq."
                )
            self.sequence_weights = sequence_weights
            self.clusters = clusters

        # If required, find indices of sequences for a subset
        if data is not None and config.input_output.subset_ids:
            self.subset = np.array([
                data.seq_ids.index(sid) for sid in config.input_output.subset_ids
            ])
        else:
            self.subset = np.arange(self.num_seq)

        # todo: Workaround
        self.effective_num_seq = self.num_seq

    def get_config(self) -> dict:
        """
        Returns a serializable configuration dictionary for this context.

        Note: Callbacks (model_lengths_cb, batch_size if callable) cannot be
        serialized directly. When deserializing, these will be reconstructed
        based on the configuration settings.
        """
        config_dict = {
            "config": self.config.model_dump(),
            "num_seq": int(self.num_seq),
            "model_lengths": self.model_lengths.tolist(),
            "sequence_weights": self.sequence_weights.tolist() if self.sequence_weights is not None else None,
            "clusters": self.clusters.tolist() if isinstance(self.clusters, np.ndarray) else self.clusters,
            "subset": self.subset.tolist(),
            "effective_num_seq": int(self.effective_num_seq),
            # Store whether batch_size is callable or int
            "batch_size_is_callable": callable(self.batch_size),
            "batch_size_value": None if callable(self.batch_size) else int(self.batch_size),
        }
        return config_dict

    @classmethod
    def from_config(cls, config_dict: dict) -> 'LearnMSAContext':
        """
        Reconstructs a LearnMSAContext from a configuration dictionary.

        Args:
            config_dict: Dictionary returned by get_config()

        Returns:
            A new LearnMSAContext instance with the same configuration.
        """
        # Reconstruct the Configuration object
        config = Configuration(**config_dict["config"])

        # Set the model lengths
        config.training.length_init = config_dict["model_lengths"]

        # Convert lists back to numpy arrays
        if config_dict["sequence_weights"] is not None:
            sequence_weights = np.array(
                config_dict["sequence_weights"], dtype=np.float32
            )
        else:
            sequence_weights = None
        clusters = config_dict["clusters"]
        if clusters is not None and isinstance(clusters, list):
            clusters = np.array(clusters)

        # Create the context (this will reconstruct callbacks)
        context = cls(
            config=config,
            num_seq=config_dict["num_seq"],
            sequence_weights=sequence_weights,
            clusters=clusters,
        )

        # Restore stored values that might differ from defaults
        context.subset = np.array(config_dict["subset"], dtype=np.int32)
        context.effective_num_seq = config_dict["effective_num_seq"]

        # Restore batch_size if it was a fixed integer
        if not config_dict["batch_size_is_callable"]:
            context.batch_size = config_dict["batch_size_value"]

        return context

    def _setup_initializers(self) -> PHMMInitializerSet:
        num_model = self.config.training.num_model
        if self.config.language_model.use_language_model:
            emission_init = [
                AminoAcidPlusMvnEmissionInitializer(
                    scoring_model_config=self.scoring_model_config,
                    num_prior_components=self.config.language_model.embedding_prior_components
                )
            for _ in range(num_model)]
            insertion_init = [
                AminoAcidPlusMvnEmissionInitializer(
                    scoring_model_config=self.scoring_model_config,
                    num_prior_components=self.config.language_model.embedding_prior_components
                )
                for _ in range(num_model)]
        else:
            emission_init = [
                initializers.make_default_emission_init()
                for _ in range(num_model)
            ]
            insertion_init = [
                initializers.make_default_insertion_init()
                for _ in range(num_model)
            ]

        transition_init = [initializers.make_default_transition_init()
                for _ in range(num_model)]
        flank_init = [initializers.make_default_flank_init()
                for _ in range(num_model)]

        return PHMMInitializerSet(
            match_emissions=emission_init,
            insert_emissions=insertion_init,
            transitions=transition_init,
            start=flank_init
        )

    def _setup_init_msa(self) -> tuple[PHMMInitializerSet, ModelLengthsCallback]:
        """Set up model initializers based on configuration."""
        from_msa = self.config.init_msa.from_msa
        if self.config.init_msa.pseudocounts:
            # Infer meaningful pseudocounts from Dirichlet priors
            aa_prior = Priors.AminoAcidPrior()
            aa_prior.build()
            aa_psc = aa_prior.emission_dirichlet_mix.make_alpha()[0].numpy()
            # Add counts for special amino acids
            aa_psc = np.pad(aa_psc, (0, 3), constant_values=1e-2)
            transition_prior = Priors.ProfileHMMTransitionPrior()
            transition_prior.build()
            match_psc = transition_prior.match_dirichlet.make_alpha()[0].numpy()
            ins_psc = transition_prior.insert_dirichlet.make_alpha()[0].numpy()
            del_psc = transition_prior.delete_dirichlet.make_alpha()[0].numpy()
            del aa_prior
            del transition_prior
        else:
            # Use very small pseudocounts to avoid zero probabilities
            aa_psc = 1e-2
            match_psc = 1e-2
            ins_psc = 1e-2
            del_psc = 1e-2

        # Load the MSA and count
        with AlignedDataset(from_msa, "fasta") as input_msa:
            values = MSA2HMM.PHMMValueSet.from_msa(
                input_msa,
                match_threshold=self.config.init_msa.match_threshold,
                global_factor=self.config.init_msa.global_factor,
            ).add_pseudocounts(
                aa=aa_psc,
                match_transition=match_psc,
                insert_transition=ins_psc,
                delete_transition=del_psc,
                begin_to_match=1e-2,
                match_to_end=1e-2,
                left_flank=ins_psc,
                right_flank=ins_psc,
                unannotated=ins_psc,
                end=1e-2,
                flank_start=ins_psc,
            ).normalize().log()

            if self.config.language_model.use_language_model:
                from learnMSA.protein_language_models.MvnEmitter import \
                    AminoAcidPlusMvnEmissionInitializer
                dim = len(input_msa.alphabet)-1 + 2 * self.scoring_model_config.dim
                emb_kernel = AminoAcidPlusMvnEmissionInitializer(
                    self.scoring_model_config
                )((1,1,1,dim)).numpy().squeeze() #type: ignore
                emb_kernel = emb_kernel[len(input_msa.alphabet)-1:]
            else:
                emb_kernel = None

            # Apply random noise only when using multiple models
            if self.config.training.num_model > 1:
                random_scale=self.config.init_msa.random_scale
            else:
                random_scale=0.0
            initializers = make_initializers_from(
                values,
                num_models=self.config.training.num_model,
                random_scale=random_scale,
                emission_kernel_extra=emb_kernel,
            )
            model_lengths_cb = lambda data: \
                np.array([values.matches()]*self.config.training.num_model)
            if self.config.input_output.verbose:
                print(
                    f"Initialized from MSA '{self.config.init_msa.from_msa}' with "
                    f"{values.matches()} match states."
                )

            return initializers, model_lengths_cb


    def _setup_lengths(self) -> ModelLengthsCallback | None:
        """Set up model lengths based on configuration."""
        # Handle length_init: if provided, update num_model and set custom callback
        length_init = self.config.training.length_init
        if length_init is not None:
            # Ensure all lengths are at least 3
            self.config.training.length_init = [
                max(3, length) for length in length_init
            ]
            # Create callback to return the specified lengths
            specified_lengths = np.array(length_init, dtype=np.int32)
            if self.config.input_output.verbose:
                print(
                    "Using user-specified initial model lengths: "\
                    f"{self.config.training.length_init}"
                )
            return lambda data: specified_lengths
        return None


    def _setup_hmm_components(self) -> tuple[list[tf.keras.Layer], tf.keras.Layer]:
        if self.config.language_model.use_language_model:
            emitter = MvnEmitter(
                self.scoring_model_config,
                emission_init=self.initializers.match_emissions,
                insertion_init=self.initializers.insert_emissions,
                num_prior_components=self.config.language_model.embedding_prior_components,
                full_covariance=False,
                temperature=self.config.language_model.temperature,
                frozen_insertions=self.config.training.frozen_insertions,
                inv_gamma_alpha=self.config.advanced.inverse_gamma_alpha,
                inv_gamma_beta=self.config.advanced.inverse_gamma_beta,
            )
        else:
            emitter = emit.ProfileHMMEmitter(
                emission_init=self.initializers.match_emissions, # type: ignore
                insertion_init=self.initializers.insert_emissions, # type: ignore
            )
        transitioner = trans.ProfileHMMTransitioner(
            transition_init = self.initializers.transitions,
            flank_init = self.initializers.start # type: ignore
        )
        return [emitter], transitioner


    def _setup_batch_size_cb(self) -> BatchSizeCallback | int:
        """ Check if a custom batch size or tokens per batch is set.
        If not, setup a callback to automatically scale the batch size based
        on sequence lengths and available GPU memory.
        """
        if self.config.training.tokens_per_batch > 0:
            def _batch_size_cb_with_tokens(data: SequenceDataset):
                return training_util.tokens_per_batch_to_batch_size(
                    self.model_lengths.tolist(),
                    min(data.max_len, int(self.config.training.crop)),
                    tokens_per_batch=self.config.training.tokens_per_batch
                )
            return _batch_size_cb_with_tokens
        elif self.config.training.batch_size > 0:
            return self.config.training.batch_size

        use_language_model = self.config.language_model.use_language_model
        #if there is at least one GPU, check its memory
        gpu_mem = get_gpu_memory()
        small_gpu = gpu_mem[0] < 32000 if len(gpu_mem) > 0 else False
        if use_language_model:
            def _batch_size_cb_with_plm(data: SequenceDataset):
                return training_util.get_adaptive_batch_size_with_language_model(
                    self.model_lengths.tolist(),
                    min(data.max_len, int(self.config.training.crop)),
                    embedding_dim=self.scoring_model_config.dim,
                    small_gpu=small_gpu,
                )
            return _batch_size_cb_with_plm
        else:
            def _batch_size_cb(data: SequenceDataset):
                return training_util.get_adaptive_batch_size(
                    self.model_lengths.tolist(),
                    min(data.max_len, int(self.config.training.crop)),
                    small_gpu=small_gpu,
                )
            return _batch_size_cb

    def _setup_visualization(self) -> None:
        """Set up visualization file paths based on configuration."""
        if self.config.visualization.logo:
            self.logo_path = validate_filepath(
                self.config.visualization.logo, ".pdf"
            )
            os.makedirs(self.logo_path.parent, exist_ok=True)
        if self.config.visualization.logo_gif:
            self.logo_gif_path = validate_filepath(
                self.config.visualization.logo_gif, ".gif"
            )
            os.makedirs(self.logo_gif_path.parent, exist_ok=True)
            os.makedirs(self.logo_gif_path.parent / "frames", exist_ok=True)

    def _setup_language_model_specific_settings(self) -> None:
        """
        Overrides some settings when using a language model.
        """
        if self.config.training.learning_rate != 0.1:
            warnings.warn(
                "A non-default learning rate is set while using a "\
                "language model. This setting is overridden to 0.05."
            )
        if self.config.training.epochs != [10, 2, 10]:
            warnings.warn(
                "A non-default number of epochs is set while using "\
                "a language model. This setting is overridden to [10, 4, 20]."
            )
        if self.config.training.cluster_seq_id != 0.9:
            warnings.warn(
                "A non-default cluster_seq_id is set while using a "
                "language model. This setting is overridden to 0.9."
            )
        self.config.training.learning_rate = 0.05
        self.config.training.epochs = [10, 4, 20]
        self.config.training.cluster_seq_id = 0.5


    def _get_scoring_model_config(self) -> Common.ScoringModelConfig:
        if self.config.language_model.use_language_model:
            scoring_model_config = Common.ScoringModelConfig(
                lm_name=self.config.language_model.language_model,
                dim=self.config.language_model.scoring_model_dim,
                activation=self.config.language_model.scoring_model_activation,
                suffix=self.config.language_model.scoring_model_suffix,
                scaled=False
            )
        else:
            scoring_model_config = Common.ScoringModelConfig()
        return scoring_model_config


    def _get_batch_generator(self) -> train.BatchGenerator:
        if self.config.language_model.use_language_model:
            batch_gen = EmbeddingBatchGenerator.EmbeddingBatchGenerator(
                scoring_model_config=self.scoring_model_config,
            )
        else:
            batch_gen = train.BatchGenerator()
        return batch_gen


    def _get_clustering(
        self, data: SequenceDataset
    ) -> tuple[np.ndarray | None, Any]:
        from ..msa_hmm import SequenceDataset
        if not self.config.training.no_sequence_weights:
            os.makedirs(self.config.input_output.work_dir, exist_ok=True)
            try:
                if self.config.input_output.input_file == Path():
                # When no input file is provided, we need to write a temporary
                # for mmseqs2 clustering
                    cluster_file = os.path.join(
                        self.config.input_output.work_dir,
                        "temp_for_clustering.fasta"
                    )
                    data.write(cluster_file, "fasta")
                elif self.config.input_output.input_format == "fasta":
                    # When the input file is fasta: all good
                    cluster_file = self.config.input_output.input_file
                else:
                    # We need to convert to fasta
                    cluster_file = os.path.join(
                        self.config.input_output.work_dir,
                        os.path.basename(self.config.input_output.input_file) + ".temp_for_clustering"
                    )
                    with SequenceDataset(
                        self.config.input_output.input_file,
                        self.config.input_output.input_format
                    ) as data:
                        data.write(cluster_file, "fasta")
                sequence_weights, clusters = clustering.compute_sequence_weights(
                    cluster_file,
                    self.config.input_output.work_dir,
                    self.config.training.cluster_seq_id,
                    return_clusters=True
                )
            except Exception as e:
                raise ValueError("Error while computing sequence weights.")
        else:
            sequence_weights, clusters = None, None
        return sequence_weights, clusters
