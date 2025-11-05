from pathlib import Path

from learnMSA.run.args import parse_args
from learnMSA.run.args_to_config import args_to_config


class TestArgsToConfig:
    """Tests for converting argparse Namespace to Configuration."""

    def test_args_to_config_with_minimal_args(self):
        """Test conversion with only required arguments."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m"
        ])

        config = args_to_config(args)

        # Check input/output config exists and has correct defaults
        assert config.input_output is not None
        assert config.input_output.input_file == Path("input.fasta")
        assert config.input_output.output_file == Path("output.a2m")
        assert config.input_output.format == "a2m"
        assert config.input_output.input_format == "fasta"
        assert config.input_output.save_model == ""
        assert config.input_output.load_model == ""
        assert config.input_output.silent is False
        assert config.input_output.cuda_visible_devices == "default"
        assert config.input_output.work_dir == "tmp"
        assert config.input_output.convert is False

        # Check training defaults are applied
        assert config.training.num_model == 4
        assert config.training.batch_size == -1
        assert config.training.tokens_per_batch == -1
        assert config.training.learning_rate == 0.1
        assert config.training.epochs == [10, 2, 10]
        assert config.training.crop == "auto"
        assert config.init_msa.from_msa is None
        assert config.language_model.use_language_model is False
        assert config.visualization.logo == ""
        assert config.advanced.alpha_flank == 7000

    def test_args_to_config_with_training_args(self):
        """Test conversion with training-related arguments."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "-n", "7",
            "--tokens_per_batch", "2000",
            "--learning_rate", "0.05",
            "--epochs", "20", "5", "15",
            "--max_iterations", "5",
            "--length_init", "10", "20", "30",
            "--length_init_quantile", "0.7",
            "--surgery_quantile", "0.6",
            "--min_surgery_seqs", "50000",
            "--len_mul", "0.9",
            "--surgery_del", "0.4",
            "--surgery_ins", "0.6",
            "--model_criterion", "BIC",
            "--indexed_data",
            "--unaligned_insertions",
            "--crop", "500",
            "--auto_crop_scale", "3.0",
            "--frozen_insertions",
            "--no_sequence_weights",
            "--skip_training",
        ])

        config = args_to_config(args)

        # num_model is overridden by length_init (3 elements)
        assert config.training.num_model == 3
        assert config.training.tokens_per_batch == 2000
        assert config.training.learning_rate == 0.05
        assert config.training.epochs == [20, 5, 15]
        assert config.training.max_iterations == 5
        assert config.training.length_init == [10, 20, 30]
        assert config.training.length_init_quantile == 0.7
        assert config.training.surgery_quantile == 0.6
        assert config.training.min_surgery_seqs == 50000
        assert config.training.len_mul == 0.9
        assert config.training.surgery_del == 0.4
        assert config.training.surgery_ins == 0.6
        assert config.training.model_criterion == "BIC"
        assert config.training.indexed_data is True
        assert config.training.unaligned_insertions is True
        assert config.training.crop == 500
        assert config.training.auto_crop_scale == 3.0
        assert config.training.frozen_insertions is True
        assert config.training.no_sequence_weights is True
        assert config.training.skip_training is True

    def test_args_to_config_with_epochs_single_value(self):
        """Test that single epoch value is expanded to 3."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--epochs", "25"
        ])

        config = args_to_config(args)
        assert config.training.epochs == [25, 25, 25]

    def test_args_to_config_with_init_msa_args(self):
        """Test conversion with init MSA arguments."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--from_msa", "/path/to/msa.fasta",
            "--match_threshold", "0.7",
            "--global_factor", "0.5",
            "--random_scale", "0.01",
            "--pseudocounts",
        ])

        config = args_to_config(args)

        assert config.init_msa.from_msa == Path("/path/to/msa.fasta")
        assert config.init_msa.match_threshold == 0.7
        assert config.init_msa.global_factor == 0.5
        assert config.init_msa.random_scale == 0.01
        assert config.init_msa.pseudocounts is True

    def test_args_to_config_with_language_model_args(self):
        """Test conversion with language model arguments."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--use_language_model",
            "--plm_cache_dir", "/models/cache",
            "--language_model", "esm2",
            "--scoring_model_dim", "32",
            "--scoring_model_activation", "relu",
            "--scoring_model_suffix", "_v2",
            "--temperature", "5.0",
            "--temperature_mode", "fixed",
            "--use_L2",
            "--L2_match", "0.5",
            "--L2_insert", "500.0",
            "--embedding_prior_components", "64",
        ])

        config = args_to_config(args)

        assert config.language_model.use_language_model is True
        assert config.language_model.plm_cache_dir == "/models/cache"
        assert config.language_model.language_model == "esm2"
        assert config.language_model.scoring_model_dim == 32
        assert config.language_model.scoring_model_activation == "relu"
        assert config.language_model.scoring_model_suffix == "_v2"
        assert config.language_model.temperature == 5.0
        assert config.language_model.temperature_mode == "fixed"
        assert config.language_model.use_L2 is True
        assert config.language_model.L2_match == 0.5
        assert config.language_model.L2_insert == 500.0
        assert config.language_model.embedding_prior_components == 64

    def test_args_to_config_with_visualization_args(self):
        """Test conversion with visualization arguments."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--logo", "output/logo.pdf",
            "--logo_gif", "output/logo.gif",
        ])

        config = args_to_config(args)

        assert config.visualization.logo == "output/logo.pdf"
        assert config.visualization.logo_gif == "output/logo.gif"

    def test_args_to_config_with_advanced_args(self):
        """Test conversion with advanced/development arguments."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--dist_out", "distributions.txt",
            "--alpha_flank", "5000",
            "--alpha_single", "1e8",
            "--alpha_global", "5000",
            "--alpha_flank_compl", "2",
            "--alpha_single_compl", "3",
            "--alpha_global_compl", "4",
            "--inverse_gamma_alpha", "2.0",
            "--inverse_gamma_beta", "1.0",
            "--frozen_distances",
            "--initial_distance", "0.1",
            "--trainable_rate_matrices",
        ])

        config = args_to_config(args)

        assert config.advanced.dist_out == "distributions.txt"
        assert config.advanced.alpha_flank == 5000
        assert config.advanced.alpha_single == 1e8
        assert config.advanced.alpha_global == 5000
        assert config.advanced.alpha_flank_compl == 2
        assert config.advanced.alpha_single_compl == 3
        assert config.advanced.alpha_global_compl == 4
        assert config.advanced.inverse_gamma_alpha == 2.0
        assert config.advanced.inverse_gamma_beta == 1.0
        assert config.advanced.frozen_distances is True
        assert config.advanced.initial_distance == 0.1
        assert config.advanced.trainable_rate_matrices is True

    def test_args_to_config_num_model_from_length_init(self):
        """Test that num_model is computed from length_init when provided."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "-n", "4",  # This will be overridden
            "--length_init", "5", "10", "15", "20", "25",  # 5 elements
        ])

        config = args_to_config(args)
        # num_model should be computed from length_init (5 elements)
        assert config.training.num_model == 5
        assert config.training.length_init == [5, 10, 15, 20, 25]

    def test_args_to_config_comprehensive(self):
        """Test conversion with a comprehensive set of arguments."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "sequences.fasta",
            "-o", "alignment.a2m",
            "-f", "stockholm",
            "-n", "3",
            "-b", "16",
            "--learning_rate", "0.02",
            "--epochs", "15", "3", "12",
            "--max_iterations", "3",
            "--length_init", "8", "16", "24",
            "--crop", "200",
            "--from_msa", "/data/input.fasta",
            "--match_threshold", "0.6",
            "--use_language_model",
            "--language_model", "proteinBERT",
            "--logo", "logo.pdf",
            "--logo_gif", "logo.gif",
            "--alpha_flank", "6000",
        ])

        config = args_to_config(args)

        # Verify configuration can be serialized
        config_dict = config.model_dump()

        assert "input_output" in config_dict
        assert "training" in config_dict
        assert "init_msa" in config_dict
        assert "language_model" in config_dict
        assert "visualization" in config_dict
        assert "advanced" in config_dict

        # Spot-check some values
        assert config.training.num_model == 3
        assert config.input_output is not None
        assert config.input_output.format == "stockholm"
        assert config.training.batch_size == 16
        assert config.training.crop == 200
        assert config.init_msa.from_msa == Path("/data/input.fasta")
        assert config.language_model.use_language_model is True
        assert config.language_model.language_model == "proteinBERT"
        assert config.visualization.logo == "logo.pdf"
        assert config.advanced.alpha_flank == 6000

    def test_args_to_config_with_all_boolean_flags(self):
        """Test conversion with all boolean flags set."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--indexed_data",
            "--unaligned_insertions",
            "--frozen_insertions",
            "--no_sequence_weights",
            "--skip_training",
            "--pseudocounts",
            "--use_language_model",
            "--use_L2",
            "--frozen_distances",
            "--trainable_rate_matrices",
        ])

        config = args_to_config(args)

        assert config.training.indexed_data is True
        assert config.training.unaligned_insertions is True
        assert config.training.frozen_insertions is True
        assert config.training.no_sequence_weights is True
        assert config.training.skip_training is True
        assert config.init_msa.pseudocounts is True
        assert config.language_model.use_language_model is True
        assert config.language_model.use_L2 is True
        assert config.advanced.frozen_distances is True
        assert config.advanced.trainable_rate_matrices is True

    def test_args_to_config_short_options(self):
        """Test conversion using short option names."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "-n", "6",
            "-b", "64",
            "-f", "fasta",
        ])

        config = args_to_config(args)

        assert config.training.num_model == 6
        assert config.training.batch_size == 64

    def test_standard_msa_with_language_model(self):
        """Test: Standard MSA with language model (recommended usage)."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--use_language_model",
        ])

        config = args_to_config(args)

        assert config.language_model.use_language_model is True
        assert config.language_model.language_model == "protT5"  # default

    def test_quick_alignment_no_surgery(self):
        """Test: Quick alignment without model surgery (max_iterations 1)."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--max_iterations", "1",
        ])

        config = args_to_config(args)

        assert config.training.max_iterations == 1

    def test_high_quality_more_models(self):
        """Test: High-quality alignment with more models and iterations."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--use_language_model",
            "-n", "10",
            "--max_iterations", "3",
        ])

        config = args_to_config(args)

        assert config.training.num_model == 10
        assert config.training.max_iterations == 3
        assert config.language_model.use_language_model is True

    def test_limited_gpu_memory(self):
        """Test: Limited GPU memory - reduce batch size and num models."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "-n", "2",
            "-b", "32",
        ])

        config = args_to_config(args)

        assert config.training.num_model == 2
        assert config.training.batch_size == 32

    def test_custom_epoch_scheme(self):
        """Test: Custom epoch scheme for different iterations."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--epochs", "20", "3", "20",
        ])

        config = args_to_config(args)

        assert config.training.epochs == [20, 3, 20]

    def test_init_from_msa(self):
        """Test: Initialize from existing MSA."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--from_msa", "/path/to/initial.fasta",
            "--max_iterations", "1",
        ])

        config = args_to_config(args)

        assert config.init_msa.from_msa == Path("/path/to/initial.fasta")
        assert config.training.max_iterations == 1

    def test_init_from_msa_minimal_finetuning(self):
        """Test: Initialize from MSA with minimal finetuning."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--from_msa", "/path/to/initial.fasta",
            "--epochs", "3",
            "--max_iterations", "1",
            "-n", "1",
        ])

        config = args_to_config(args)

        assert config.init_msa.from_msa == Path("/path/to/initial.fasta")
        assert config.training.epochs == [3, 3, 3]
        assert config.training.max_iterations == 1
        assert config.training.num_model == 1

    def test_skip_training_with_loaded_model(self):
        """Test: Skip training and use pre-trained model."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--skip_training",
        ])

        config = args_to_config(args)

        assert config.training.skip_training is True

    def test_different_language_model(self):
        """Test: Use different language model (esm2)."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--use_language_model",
            "--language_model", "esm2",
        ])

        config = args_to_config(args)

        assert config.language_model.use_language_model is True
        assert config.language_model.language_model == "esm2"

    def test_custom_plm_cache(self):
        """Test: Custom PLM cache directory."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--use_language_model",
            "--plm_cache_dir", "/custom/cache",
            "--language_model", "protT5",
        ])

        config = args_to_config(args)

        assert config.language_model.use_language_model is True
        assert config.language_model.plm_cache_dir == "/custom/cache"
        assert config.language_model.language_model == "protT5"

    def test_visualization_logo(self):
        """Test: Generate sequence logo."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--logo", "output/logo.pdf",
        ])

        config = args_to_config(args)

        assert config.visualization.logo == "output/logo.pdf"

    def test_visualization_logo_gif(self):
        """Test: Generate animated sequence logo."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--logo_gif", "output/logo_animation.gif",
        ])

        config = args_to_config(args)

        assert config.visualization.logo_gif == "output/logo_animation.gif"

    def test_custom_match_threshold(self):
        """Test: Customize match threshold for MSA initialization."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--from_msa", "/path/to/initial.fasta",
            "--match_threshold", "0.7",
            "--max_iterations", "1",
        ])

        config = args_to_config(args)

        assert config.init_msa.from_msa == Path("/path/to/initial.fasta")
        assert config.init_msa.match_threshold == 0.7
        assert config.training.max_iterations == 1

    def test_pseudocounts_for_small_msa(self):
        """Test: Add pseudocounts when initializing from small MSA."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--from_msa", "/path/to/small_msa.fasta",
            "--pseudocounts",
        ])

        config = args_to_config(args)

        assert config.init_msa.from_msa == Path("/path/to/small_msa.fasta")
        assert config.init_msa.pseudocounts is True

    def test_crop(self):
        """Test: Memory optimization with crop and indexed data."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--crop", "1000",
        ])

        config = args_to_config(args)

        assert config.training.crop == 1000

    def test_disable_crop(self):
        """Test: Disable sequence cropping."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--crop", "disable",
        ])

        config = args_to_config(args)

        assert config.training.crop == "disable"

    def test_auto_crop_with_scale(self):
        """Test: Auto crop with custom scale factor."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--crop", "auto",
            "--auto_crop_scale", "3.0",
        ])

        config = args_to_config(args)

        assert config.training.crop == "auto"
        assert config.training.auto_crop_scale == 3.0

    def test_save_model(self):
        """Test: Save trained model for later reuse."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--save_model", "my_model",
        ])

        config = args_to_config(args)

        # save_model is now part of InputOutputConfig
        assert config.input_output is not None
        assert config.input_output.save_model == "my_model"

    def test_load_model_with_skip_training(self):
        """Test: Load pre-trained model and skip training."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--load_model", "my_model",
            "--skip_training",
        ])

        config = args_to_config(args)

        # load_model is now part of InputOutputConfig
        assert config.input_output is not None
        assert config.input_output.load_model == "my_model"
        assert config.training.skip_training is True

    def test_load_model_with_continued_training(self):
        """Test: Load pre-trained model and continue training."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
            "--load_model", "my_model",
            "--epochs", "5",
        ])

        config = args_to_config(args)

        # load_model is now part of InputOutputConfig
        assert config.input_output is not None
        assert config.input_output.load_model == "my_model"
        assert config.training.skip_training is False
        assert config.training.epochs == [5, 5, 5]

    def test_save_and_load_model_default_empty(self):
        """Test: save_model and load_model default to empty strings."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
        ])

        config = args_to_config(args)

        # Both should default to empty strings in InputOutputConfig
        assert config.input_output is not None
        assert config.input_output.save_model == ""
        assert config.input_output.load_model == ""

    def test_convert_a2m_to_fasta(self):
        """Test: Convert MSA format from a2m to fasta."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "proteins.a2m",
            "-o", "proteins.fasta",
            "--convert",
            "-f", "fasta",
        ])

        config = args_to_config(args)

        # convert is now part of InputOutputConfig
        assert config.input_output is not None
        assert config.input_output.convert is True
        assert config.input_output.format == "fasta"
        assert config.input_output.input_file == Path("proteins.a2m")
        assert config.input_output.output_file == Path("proteins.fasta")

        # When converting, training parameters still get defaults
        assert config.training.num_model == 4

    def test_convert_defaults_to_false(self):
        """Test: convert flag defaults to False."""
        parser = parse_args("test_version")
        args = parser.parse_args([
            "-i", "input.fasta",
            "-o", "output.a2m",
        ])

        config = args_to_config(args)

        assert config.input_output is not None
        assert config.input_output.convert is False

