import numpy as np
import pytest
import tensorflow as tf
from pathlib import Path
from typing import Generator

from learnMSA import Configuration
from learnMSA.config import (InitMSAConfig, InputOutputConfig,
                             LanguageModelConfig, TrainingConfig)
from learnMSA.msa_hmm.learnmsa_context import LearnMSAContext
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset

DIR = "tests/data/"

# Test the context with a dataset

@pytest.fixture
def simple_data() -> Generator[SequenceDataset, None, None]:
    """Fixture providing a small sequence dataset."""
    with SequenceDataset(f"{DIR}/felix.fa", "fasta") as data:
        yield data

@pytest.fixture
def config() -> Configuration:
    return Configuration(
        input_output=InputOutputConfig(verbose=False),
        training=TrainingConfig(no_sequence_weights=True)
    )

def test_basic_initialization(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test basic context initialization with dataset."""
    context = LearnMSAContext(config=config, data=simple_data)

    assert context.num_seq == simple_data.num_seq
    assert context.config == config
    assert isinstance(context.model_lengths, np.ndarray)
    assert len(context.model_lengths) == config.training.num_model
    assert context.sequence_weights is not None or config.training.no_sequence_weights
    assert context.subset.shape[0] == simple_data.num_seq

def test_model_lengths_callback(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test that model_lengths_cb is properly set and callable."""
    context = LearnMSAContext(config=config, data=simple_data)

    assert callable(context.model_lengths_cb)
    # Should be able to call it again with the same data
    new_lengths = context.model_lengths_cb(simple_data)
    assert isinstance(new_lengths, np.ndarray)
    assert len(new_lengths) == config.training.num_model

def test_with_specified_length_init(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test initialization with user-specified model lengths."""
    lengths = [10, 15, 20]
    config.training.length_init = lengths
    context = LearnMSAContext(config=config, data=simple_data)

    assert context.config.training.num_model == 3
    np.testing.assert_array_equal(context.model_lengths, lengths)

def test_batch_size_callback(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test batch size setup with default adaptive sizing."""
    context = LearnMSAContext(config=config, data=simple_data)

    # Default should be a callable
    assert callable(context.batch_size)
    batch_size = context.batch_size(simple_data)
    assert isinstance(batch_size, int)
    assert batch_size > 0

def test_batch_size_fixed(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test batch size with fixed value."""
    fixed_batch = 64
    config.training.batch_size = fixed_batch
    context = LearnMSAContext(config=config, data=simple_data)

    assert context.batch_size == fixed_batch

def test_batch_size_tokens_per_batch(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test batch size with tokens_per_batch setting."""
    config.training.tokens_per_batch = 10000
    context = LearnMSAContext(config=config, data=simple_data)

    assert callable(context.batch_size)
    batch_size = context.batch_size(simple_data)
    assert isinstance(batch_size, int)
    assert batch_size > 0

def test_auto_crop_setup(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test automatic cropping length setup."""
    config.training.auto_crop = True
    config.training.auto_crop_scale = 1.5
    context = LearnMSAContext(config=config, data=simple_data)

    expected_crop = int(np.ceil(1.5 * np.mean(simple_data.seq_lens)))
    assert context.config.training.crop == expected_crop

def test_emitter_and_transitioner_setup(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test that emitter and transitioner are properly initialized."""
    context = LearnMSAContext(config=config, data=simple_data)

    assert isinstance(context.emitter, list)
    assert len(context.emitter) > 0
    assert all(isinstance(e, tf.keras.layers.Layer) for e in context.emitter)
    assert isinstance(context.transitioner, tf.keras.layers.Layer)

def test_encoder_initializer_setup(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test encoder initializer is properly configured."""
    context = LearnMSAContext(config=config, data=simple_data)

    assert context.encoder_initializer is not None
    assert len(context.encoder_initializer) == 3

def test_with_subset_ids(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test context creation with subset of sequences."""
    # Get some sequence IDs from the dataset
    subset_ids = [simple_data.seq_ids[i] for i in [0, 2, 3]]
    config.input_output.subset_ids = subset_ids
    context = LearnMSAContext(config=config, data=simple_data)

    assert len(context.subset) == len(subset_ids)
    # Verify the indices correspond to the correct IDs
    for idx, seq_id in zip(context.subset, subset_ids):
        assert simple_data.seq_ids[idx] == seq_id

def test_skip_training_mode(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test that skip_training adjusts epochs correctly."""
    config.training.skip_training = True
    context = LearnMSAContext(config=config, data=simple_data)

    assert context.config.training.max_iterations == 1
    assert context.config.training.epochs == [0, 0, 0]

def test_sequence_weights_no_weights(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test with sequence weights disabled."""
    context = LearnMSAContext(config=config, data=simple_data)

    assert context.sequence_weights is None
    assert context.clusters is None


# Test the context without a dataset

def test_basic_initialization_without_data(config: Configuration) -> None:
    """Test basic context initialization without dataset."""
    num_seq = 100
    lengths = [10, 15, 20]
    config.training.length_init = lengths
    context = LearnMSAContext(config=config,num_seq=num_seq)

    assert context.num_seq == num_seq
    assert context.config == config
    np.testing.assert_array_equal(context.model_lengths, lengths)

def test_requires_num_seq_when_no_data(config: Configuration) -> None:
    """Test that num_seq is required when data is None."""
    config.training.length_init = [10, 20]

    with pytest.raises(AssertionError, match="num_seq must be specified"):
        LearnMSAContext(config=config, data=None)

def test_requires_length_init_when_no_data(config):
    """Test that length_init is required when data is None."""
    with pytest.raises(AssertionError, match="length_init must be specified"):
        LearnMSAContext(config=config, data=None, num_seq=100)

def test_with_custom_sequence_weights(config: Configuration) -> None:
    """Test initialization with custom sequence weights."""
    num_seq = 50
    weights = np.random.rand(num_seq).astype(np.float32)
    config.training.length_init = [10, 15]
    context = LearnMSAContext(
        config=config,
        num_seq=num_seq,
        sequence_weights=weights
    )

    assert context.sequence_weights is not None
    np.testing.assert_array_equal(context.sequence_weights, weights)

def test_default_sequence_weights_ones(config: Configuration) -> None:
    """Test that default sequence weights are all ones."""
    num_seq = 50
    config.training.length_init = [10, 15]
    context = LearnMSAContext(
        config=config,
        num_seq=num_seq
    )

    assert context.sequence_weights is not None
    np.testing.assert_array_equal(
        context.sequence_weights,
        np.ones(num_seq, dtype=np.float32)
    )

def test_with_custom_clusters(config: Configuration) -> None:
    """Test initialization with custom cluster assignments."""
    num_seq = 50
    clusters = np.random.randint(0, 5, size=num_seq)
    config.training.length_init = [10, 15]
    context = LearnMSAContext(
        config=config,
        num_seq=num_seq,
        clusters=clusters
    )

    np.testing.assert_array_equal(context.clusters, clusters)

def test_sequence_weights_length_validation(config: Configuration) -> None:
    """Test that sequence_weights length must match num_seq."""
    num_seq = 50
    wrong_weights = np.random.rand(30).astype(np.float32)
    config.training.length_init = [10, 15]

    with pytest.raises(AssertionError, match="does not match num_seq"):
        LearnMSAContext(
            config=config,
            num_seq=num_seq,
            sequence_weights=wrong_weights
        )

def test_clusters_length_validation(config: Configuration) -> None:
    """Test that clusters length must match num_seq."""
    num_seq = 50
    wrong_clusters = np.random.randint(0, 5, size=30)
    config.training.length_init = [10, 15]

    with pytest.raises(AssertionError, match="does not match num_seq"):
        LearnMSAContext(
            config=config,
            num_seq=num_seq,
            clusters=wrong_clusters
        )

def test_subset_without_data(config: Configuration) -> None:
    """Test that subset is all indices when initialized without data."""
    num_seq = 50
    config.training.length_init = [10, 15]
    context = LearnMSAContext(
        config=config,
        num_seq=num_seq
    )

    np.testing.assert_array_equal(context.subset, np.arange(num_seq))

def test_num_seq_warning_when_data_provided(tmp_path, config, capsys):
    """Test warning when num_seq is provided with data."""
    # Create a temporary fasta file
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(">seq1\nACGT\n>seq2\nTGCA\n")

    with SequenceDataset(str(fasta_file), "fasta") as data:
        context = LearnMSAContext(
            config=config,
            data=data,
            num_seq=100  # Will be ignored
        )
        # Should use data.num_seq, not the provided num_seq
        assert context.num_seq == data.num_seq

        # Check that warning was printed
        captured = capsys.readouterr()
        assert "num_seq is provided but data is not None" in captured.out

@pytest.fixture
def aligned_msa(tmp_path) -> str:
    """Create a temporary aligned MSA file."""
    msa_file = tmp_path / "test_msa.fasta"
    msa_content = """>seq1\nAC-GT\n>seq2\nACGGT\n>seq3\nA--GT\n"""
    msa_file.write_text(msa_content)
    return str(msa_file)

def test_init_from_msa(
    aligned_msa: str, tmp_path: Path, config: Configuration
) -> None:
    """Test initialization from an aligned MSA."""
    # Create a simple dataset
    fasta_file = tmp_path / "seqs.fasta"
    fasta_file.write_text(">seq1\nACGT\n>seq2\nACGGT\n>seq3\nAGT\n")

    config.init_msa.from_msa = Path(aligned_msa)

    with SequenceDataset(str(fasta_file), "fasta") as data:
        context = LearnMSAContext(config=config, data=data)

        # Should have initializers set from MSA
        assert context.initializers is not None
        assert context.model_lengths_cb is not None

        # Model length should match MSA structure
        lengths = context.model_lengths_cb(data)
        assert all(length >= 3 for length in lengths)


# Batch Generator Tests
def test_default_batch_generator(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test default batch generator setup."""
    context = LearnMSAContext(config=config, data=simple_data)

    from learnMSA.msa_hmm import training
    assert isinstance(context.batch_gen, training.BatchGenerator)


def test_batch_generator_without_data(config: Configuration) -> None:
    """Test batch generator setup without dataset."""
    config.training.length_init = [10, 15]
    context = LearnMSAContext(
        config=config,
        num_seq=50
    )

    from learnMSA.msa_hmm import training
    assert isinstance(context.batch_gen, training.BatchGenerator)


# Transitioner Tests
def test_transitioner_prior_parameters(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test that transitioner prior parameters are set correctly."""
    config.advanced.alpha_flank = 2.0
    config.advanced.alpha_single = 3.0
    config.advanced.alpha_global = 4.0
    config.advanced.alpha_flank_compl = 5.0
    config.advanced.alpha_single_compl = 6.0
    config.advanced.alpha_global_compl = 7.0
    context = LearnMSAContext(config=config, data=simple_data)

    # Check transitioner prior parameters
    assert context.transitioner.prior.alpha_flank == 2.0
    assert context.transitioner.prior.alpha_single == 3.0
    assert context.transitioner.prior.alpha_global == 4.0
    assert context.transitioner.prior.alpha_flank_compl == 5.0
    assert context.transitioner.prior.alpha_single_compl == 6.0
    assert context.transitioner.prior.alpha_global_compl == 7.0


# Serialization Tests
def test_serialization_with_data(
    simple_data: SequenceDataset, config: Configuration
) -> None:
    """Test that context can be serialized and deserialized with data."""
    context = LearnMSAContext(config=config, data=simple_data)

    # Serialize
    config_dict = context.get_config()

    # Verify it's a dict with expected keys
    assert isinstance(config_dict, dict)
    assert "config" in config_dict
    assert "num_seq" in config_dict
    assert "model_lengths" in config_dict

    # Deserialize
    restored_context = LearnMSAContext.from_config(config_dict)

    # Check key attributes match
    assert restored_context.num_seq == context.num_seq
    np.testing.assert_array_equal(
        restored_context.model_lengths, context.model_lengths
    )
    np.testing.assert_array_equal(
        restored_context.subset, context.subset
    )
    assert restored_context.effective_num_seq == context.effective_num_seq


def test_serialization_without_data(config: Configuration) -> None:
    """Test serialization when initialized without data."""
    config.training.length_init = [10, 15]
    sequence_weights = np.array([1.0, 0.8, 0.5, 1.0, 0.9], dtype=np.float32)
    clusters = np.array([0, 0, 1, 1, 2])

    context = LearnMSAContext(
        config=config,
        num_seq=5,
        sequence_weights=sequence_weights,
        clusters=clusters
    )

    # Serialize
    config_dict = context.get_config()

    # Deserialize
    restored_context = LearnMSAContext.from_config(config_dict)

    # Verify everything matches
    assert restored_context.num_seq == 5
    np.testing.assert_array_equal(restored_context.model_lengths, context.model_lengths)
    assert restored_context.sequence_weights is not None
    np.testing.assert_array_almost_equal(restored_context.sequence_weights, sequence_weights)
    np.testing.assert_array_equal(restored_context.clusters, clusters)
    np.testing.assert_array_equal(restored_context.subset, np.arange(5))


def test_serialization_with_fixed_batch_size(config : Configuration) -> None:
    """Test that fixed batch size is preserved during serialization."""
    config.training.length_init = [10, 15]
    config.training.batch_size = 32
    context = LearnMSAContext(config=config, num_seq=50)

    # Serialize and deserialize
    config_dict = context.get_config()
    restored_context = LearnMSAContext.from_config(config_dict)

    # Check batch size is preserved
    assert restored_context.batch_size == 32
    assert not callable(restored_context.batch_size)


def test_serialization_with_callable_batch_size(config: Configuration) -> None:
    """Test that callable batch size is reconstructed during serialization."""
    long_seqs = [
        ("seq1", "ACGTACGTACGT"),
        ("seq2", "ACGTACGTACGT"*100),
    ]
    long_data = SequenceDataset(sequences=long_seqs)
    config.training.auto_crop = False
    config.training.crop = 150
    config.training.length_init = [5]
    context = LearnMSAContext(config=config, data=long_data)

    # Should have a callable batch size (adaptive)
    assert callable(context.batch_size)

    # Should respect cropping (max sequence length is > 150)
    # We should map to the biggest batch size
    assert context.batch_size(long_data) == 256

    # Serialize and deserialize
    config_dict = context.get_config()
    restored_context = LearnMSAContext.from_config(config_dict)

    # Should still be callable
    assert callable(restored_context.batch_size)


def test_serialization_preserves_config(config : Configuration) -> None:
    """Test that configuration is fully preserved during serialization."""
    config.advanced.alpha_flank = 2.5
    config.advanced.alpha_single = 3.5
    config.training.length_init = [12, 18]
    config.training.learning_rate = 0.05
    context = LearnMSAContext(config=config, num_seq=30)

    # Serialize and deserialize
    config_dict = context.get_config()
    restored_context = LearnMSAContext.from_config(config_dict)

    # Check configuration values
    assert restored_context.config.advanced.alpha_flank == 2.5
    assert restored_context.config.advanced.alpha_single == 3.5
    assert restored_context.config.training.learning_rate == 0.05
    assert restored_context.config.training.length_init == [12, 18]
