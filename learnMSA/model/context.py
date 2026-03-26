import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

import learnMSA.model.tf.training as train
import learnMSA.model.training_util as training_util
import learnMSA.tree.tf.initializer as initializers
from learnMSA import Configuration
from learnMSA.hmm.tf.prior import TFPHMMTransitionPrior
from learnMSA.hmm.tf.util import load_dirichlet
from learnMSA.hmm.util.value_set import PHMMValueSet
from learnMSA.run.util import validate_filepath
from learnMSA.util import clustering

from ..tree.tf.initializer import ConstantInitializer
from ..tree.tf.util import inverse_softplus
from ..util.aligned_dataset import AlignedDataset
from ..util.sequence_dataset import SequenceDataset

# Type alias for model length callback
ModelLengthsCallback = Callable[[SequenceDataset], np.ndarray]
BatchSizeCallback = Callable[[SequenceDataset], int]


class LearnMSAContext:
    """
    Sets up data-dependent context for learning a profile HMM.

    Either a SequenceDataset must be provided or the number of sequences along
    with a configuration that includes initial lengths must be specified.
    A context that is created with a SequenceDataset will be independent of
    the dataset after initialization. in the sense that the dataset is not
    stored in the context and all relevant properties depending the statistics
    of the dataset can be serialized without requiring the original dataset.

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
    batch_size: int | Callable[[SequenceDataset], int]
    batch_gen: train.BatchGenerator
    last_runtime_batch_size: int | None
    sequence_weights: np.ndarray | None
    clusters: Any
    subset: np.ndarray
    init_msa_values: Sequence[PHMMValueSet] | None

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
            data: SequenceDataset containing the sequences to align. Must be
                provided unless num_seq is specified.
            num_seq: Number of sequences for which this context is created.
                Must only be provided when data is None.
            sequence_weights: Array of sequence weights that can optionally be
                provided when data is None. If not provided, sequence weights
                will be 1.
            clusters: Cluster array with the same length as num_seq. Can be
                provided when data is None.
        """
        self.config = config
        self.init_msa_values = None

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
                print(
                    "Warning: num_seq is provided but data is not None. "
                    "It will be ignored."
                )
            self.num_seq = data.num_seq

        model_len_cb = None

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

        # Set up initializers
        if self.config.init_msa.from_msa is not None:
            model_len_cb, self.init_msa_values = self._setup_init_msa()
        self._setup_visualization()

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

        # Adjust training settings automatically if skip_training is set
        if self.config.training.skip_training:
            self.config.training.max_iterations = 1
            self.config.training.epochs = [0]*3

        self.batch_gen = train.BatchGenerator()
        self.last_runtime_batch_size = None
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
        # Use model_dump with mode='json' to ensure Path objects are serialized as strings
        config_dict = {
            "config": self.config.model_dump(mode='json'),
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
            config_dict: Dictionary returned by get_config() or wrapped by Keras

        Returns:
            A new LearnMSAContext instance with the same configuration.
        """
        # When called by Keras, config_dict is wrapped with metadata
        # Extract the actual config if it's wrapped
        if 'config' in config_dict and 'module' in config_dict:
            # This is a Keras-wrapped config, extract the inner config
            actual_config = config_dict['config']
        else:
            # This is a direct config from get_config()
            actual_config = config_dict

        # Reconstruct the Configuration object
        config = Configuration(**actual_config["config"])

        # Set the model lengths
        config.training.length_init = actual_config["model_lengths"]

        # Convert lists back to numpy arrays
        if actual_config["sequence_weights"] is not None:
            sequence_weights = np.array(
                actual_config["sequence_weights"], dtype=np.float32
            )
        else:
            sequence_weights = None
        clusters = actual_config["clusters"]
        if clusters is not None and isinstance(clusters, list):
            clusters = np.array(clusters)

        # Create the context (this will reconstruct callbacks)
        context = cls(
            config=config,
            num_seq=actual_config["num_seq"],
            sequence_weights=sequence_weights,
            clusters=clusters,
        )

        # Restore stored values that might differ from defaults
        context.subset = np.array(actual_config["subset"], dtype=np.int32)
        context.effective_num_seq = actual_config["effective_num_seq"]

        # Restore batch_size if it was a fixed integer
        if not actual_config["batch_size_is_callable"]:
            context.batch_size = actual_config["batch_size_value"]

        return context

    def _setup_init_msa(
        self
    ) -> tuple[ModelLengthsCallback, Sequence[PHMMValueSet]]:
        """Set up model initializers based on configuration."""
        from_msa = self.config.init_msa.from_msa
        if self.config.init_msa.pseudocounts:
            # Infer meaningful pseudocounts from Dirichlet priors
            # Get amino acid pseudocounts
            aa_prior = load_dirichlet(
                "amino_acid_dirichlet.weights",
                dim = len(SequenceDataset._default_alphabet)-1
            )
            aa_psc = aa_prior.matrix()[0, 0].numpy()

            # Get transition pseudocounts
            transition_prior = TFPHMMTransitionPrior(
                [5], self.config.hmm_prior
            )
            match_psc = transition_prior.match_prior.matrix()[0,0].numpy()
            ins_psc = transition_prior.insert_prior.matrix()[0,0].numpy()
            del_psc = transition_prior.delete_prior.matrix()[0,0].numpy()

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
            values = PHMMValueSet.from_msa(
                input_msa,
                match_threshold=self.config.init_msa.match_threshold,
                global_factor=self.config.init_msa.global_factor,
            ).add_pseudocounts(
                aa=aa_psc,
                match_transition=match_psc,
                insert_transition=ins_psc,
                delete_transition=del_psc,
                begin_to_match=1e-2,
                begin_to_delete=1e-8,
                match_to_end=1e-2,
                left_flank=ins_psc,
                right_flank=ins_psc,
                unannotated=ins_psc,
                end=1e-2,
                flank_start=ins_psc,
            ).normalize()

        n = self.config.training.num_model
        model_lengths_cb = lambda data: \
            np.array([values.matches()]*n)
        if self.config.input_output.verbose:
            print(
                f"Initialized from MSA '{self.config.init_msa.from_msa}' with "
                f"{values.matches()} match states."
            )

        model_values = [values] * n

        if self.config.training.use_noise and self.config.input_output.verbose:
            print(
                "Perturbing initial HMM parameters with Dirichlet noise."
                " To disable this, use the --no_noise flag."
            )

        return model_lengths_cb, model_values


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
            return lambda data: specified_lengths
        return None


    def _setup_batch_size_cb(self) -> BatchSizeCallback | int:
        """ Check if a custom batch size or tokens per batch is set.
        If not, setup a callback to automatically scale the batch size based
        on sequence lengths and available GPU memory.
        """
        use_language_model = self.config.language_model.use_language_model

        if self.config.training.tokens_per_batch > 0:
            def _batch_size_cb_with_tokens(data: SequenceDataset):
                seq_len = min(data.max_len, int(self.config.training.crop)) + 1
                return training_util.tokens_per_batch_to_batch_size(
                    tokens_per_batch=self.config.training.tokens_per_batch,
                    seq_len=seq_len,
                    impl_factor=self._get_impl_factor(),
                )
            return _batch_size_cb_with_tokens

        elif self.config.training.batch_size > 0:
            return self.config.training.batch_size

        else:
            #if there is at least one GPU, check its memory
            def _batch_size_cb(data: SequenceDataset):
                seq_len = min(data.max_len, int(self.config.training.crop)) + 1
                return training_util.get_adaptive_batch_size(
                    model_len=self.model_lengths.max(),
                    num_model=self.config.training.num_model,
                    seq_len=seq_len,
                    impl_factor=self._get_impl_factor(),
                )
            return _batch_size_cb

    def _get_impl_factor(self, inference: bool = False) -> float:
        """Get implementation factor for batch size scaling based on model
        type."""
        # Base implementation factor is smaller for inference, because we
        # don't need gradient (roughly halfes the memory usage).
        impl_factor = 0.5 if inference else 1.0

        # Increase the implementation factor (smaller batches) when pLMs are
        # used. The factor need to be especially high for inference, because
        # the batch size limit for inference is larger than for training.
        if self.config.language_model.use_language_model:
            impl_factor += 7.5 if inference else 2.0

        if self.config.structure.use_structure:
            impl_factor += 0.35 if inference else 0.7

        return impl_factor

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
            print(
                "Warning: A non-default learning rate is set while using a "
                "language model. This setting is overridden to 0.05."
            )
        if self.config.training.epochs != [10, 2, 10]:
            print(
                "Warning: A non-default number of epochs is set while using "
                "a language model. This setting is overridden to [10, 4, 20]."
            )
        if self.config.training.cluster_seq_id != 0.9:
            print(
                "Warning: A non-default cluster_seq_id is set while using a "
                "language model. This setting is overridden to 0.9."
            )
        self.config.training.learning_rate = 0.05
        self.config.training.epochs = [10, 4, 20]
        self.config.training.cluster_seq_id = 0.5
        self.config.training.trainable_insertions = False


    def _get_clustering(
        self, data: SequenceDataset
    ) -> tuple[np.ndarray | None, Any]:
        from ..util import SequenceDataset
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
                        os.path.basename(self.config.input_output.input_file)\
                            + ".temp_for_clustering"
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
                print(f"Error while computing sequence weights: {e}")
                raise ValueError(
                    "Error while computing sequence weights."
                ) from e
        else:
            sequence_weights, clusters = None, None
        return sequence_weights, clusters
