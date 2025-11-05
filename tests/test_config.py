import pytest
from pathlib import Path
from pydantic import ValidationError

from learnMSA.config import (AdvancedConfig, Configuration, HMMConfig,
                             InitMSAConfig, InputOutputConfig, LanguageModelConfig,
                             TrainingConfig, VisualizationConfig, get_value)


class TestConfiguration:
    """Tests for the main Configuration class."""

    def test_default_configuration(self):
        """Test that Configuration can be created with defaults."""
        config = Configuration()
        assert config.num_model == 4
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.init_msa, InitMSAConfig)
        assert isinstance(config.language_model, LanguageModelConfig)
        assert isinstance(config.visualization, VisualizationConfig)
        assert isinstance(config.advanced, AdvancedConfig)

    def test_num_model_getter_from_length_init(self):
        """Test num_model returns len(length_init) when set."""
        config = Configuration(
            training=TrainingConfig(length_init=[5, 10, 15])
        )
        assert config.num_model == 3

    def test_num_model_setter(self):
        """Test that num_model can be set directly."""
        config = Configuration()
        config.num_model = 10
        assert config.num_model == 10

    def test_num_model_setter_validation(self):
        """Test that num_model validates value >= 1."""
        config = Configuration()
        with pytest.raises(
            ValueError, match="num_model must be greater than or equal to 1"
        ):
            config.num_model = 0
        with pytest.raises(
            ValueError, match="num_model must be greater than or equal to 1"
        ):
            config.num_model = -5

    def test_num_model_initialization_with_alias(self):
        """Test that num_model can be set during initialization."""
        config = Configuration(num_model=7)
        assert config.num_model == 7

    def test_num_model_length_init_precedence(self):
        """Test that length_init takes precedence over num_model."""
        config = Configuration(
            num_model=10,
            training=TrainingConfig(length_init=[5, 10])
        )
        # length_init has 2 elements, so num_model should be 2
        assert config.num_model == 2

        # Setting num_model shouldn't change the result while length_init is set
        config.num_model = 20
        assert config.num_model == 2  # Still returns len(length_init)

    def test_num_model_after_clearing_length_init(self):
        """Test num_model behavior when length_init is set then cleared."""
        config = Configuration(
            num_model=5,
            training=TrainingConfig(length_init=[10, 20])
        )
        assert config.num_model == 2  # From length_init

        # Clear length_init
        config.training.length_init = None
        # Should now use the _num_model value (which was set to 5 during init)
        assert config.num_model == 5

        # Set a different length_init
        config.training.length_init = [1, 2, 3]
        assert config.num_model == 3


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        assert config.batch_size == -1
        assert config.learning_rate == 0.1
        assert config.epochs == [10, 2, 10]
        assert config.max_iterations == 2
        assert config.length_init is None

    def test_training_config_length_init(self):
        """Test TrainingConfig with length_init."""
        config = TrainingConfig(length_init=[5, 10, 15])
        assert config.length_init == [5, 10, 15]

    def test_training_config_length_init_validation(self):
        """Test that length_init validates minimum value."""
        with pytest.raises(ValidationError):
            TrainingConfig(length_init=[5, 2, 10])  # 2 is less than 3

    def test_training_config_epochs_expansion(self):
        """Test that a single epoch value is expanded to 3."""
        config = TrainingConfig(epochs=5)  # type: ignore
        assert config.epochs == [5, 5, 5]

    def test_training_config_epochs_validation(self):
        """Test that epochs must have exactly 3 elements."""
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=[10, 20])  # Only 2 elements

    def test_training_config_learning_rate_validation(self):
        """Test that learning_rate must be positive."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0)
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=-0.1)

    def test_training_config_crop_validation(self):
        """Test crop parameter validation."""
        config1 = TrainingConfig(crop="auto")
        assert config1.crop == "auto"

        config2 = TrainingConfig(crop="disable")
        assert config2.crop == "disable"

        config3 = TrainingConfig(crop=100)
        assert config3.crop == 100

        with pytest.raises(ValidationError):
            TrainingConfig(crop=0)  # Must be > 0

        with pytest.raises(ValidationError):
            TrainingConfig(crop="invalid")  # Invalid string


class TestConfigurationSerialization:
    """Tests for configuration serialization/deserialization."""

    def test_configuration_to_dict(self):
        """Test that Configuration can be serialized to dict."""
        config = Configuration(num_model=5)
        config_dict = config.model_dump()
        assert "training" in config_dict
        assert "init_msa" in config_dict
        assert "language_model" in config_dict
        assert "visualization" in config_dict
        assert "advanced" in config_dict
        assert "num_model" in config_dict
        assert config_dict["num_model"] == 5

    def test_configuration_from_dict(self):
        """Test that Configuration can be created from dict."""
        config_dict = {
            "num_model": 7,
            "training": {"learning_rate": 0.05}
        }
        config = Configuration(**config_dict)
        assert config.num_model == 7
        assert config.training.learning_rate == 0.05

    def test_configuration_roundtrip_with_num_model(self):
        """Test that num_model is preserved through serialization."""
        config = Configuration(
            num_model=10,
            training=TrainingConfig(learning_rate=0.01)
        )
        assert config.num_model == 10

        # Serialize and deserialize - num_model is now included
        config_dict = config.model_dump()
        assert config_dict["num_model"] == 10

        config2 = Configuration(**config_dict)
        assert config2.num_model == 10
        assert config2.training.learning_rate == 0.01

    def test_configuration_with_length_init_serialization(self):
        """Test serialization when length_init is set."""
        config = Configuration(
            training=TrainingConfig(length_init=[10, 20, 30])
        )
        config_dict = config.model_dump()
        # num_model is computed from length_init and included in serialization
        assert config.num_model == 3
        assert config_dict["num_model"] == 3
        assert config_dict["training"]["length_init"] == [10, 20, 30]


# class TestIntegration:
#     """Integration tests for the configuration system."""

#     def test_full_configuration_workflow(self):
#         """Test a complete configuration workflow."""
#         # Create a configuration
#         config = Configuration(
#             num_model=5,
#             training=TrainingConfig(
#                 learning_rate=0.05,
#                 epochs=20,  # type: ignore
#                 max_iterations=5
#             ),
#             visualization=VisualizationConfig(),
#             advanced=AdvancedConfig()
#         )

#         # Verify initial state
#         assert config.num_model == 5
#         assert config.training.epochs == [20, 20, 20]

#         # Modify training config to add length_init
#         config.training.length_init = [10, 20, 30]

#         # num_model should now reflect length_init
#         assert config.num_model == 3

#         # Serialize and deserialize
#         config_dict = config.model_dump()
#         config2 = Configuration(**config_dict)

#         # Verify deserialized config
#         assert config2.num_model == 3
#         assert config2.training.length_init == [10, 20, 30]


class TestInitMSAConfig:
    """Tests for InitMSAConfig."""

    def test_init_msa_config_defaults(self):
        """Test InitMSAConfig default values."""
        config = InitMSAConfig()
        assert config.from_msa is None
        assert config.match_threshold == 0.5
        assert config.global_factor == 0.1
        assert config.random_scale == 1e-3
        assert config.pseudocounts is False

    def test_init_msa_config_with_path(self):
        """Test InitMSAConfig with a file path."""
        from pathlib import Path
        config = InitMSAConfig(from_msa=Path("/path/to/msa.fasta"))
        assert config.from_msa == Path("/path/to/msa.fasta")

    def test_init_msa_config_custom_values(self):
        """Test InitMSAConfig with custom values."""
        config = InitMSAConfig(
            match_threshold=0.7,
            global_factor=0.5,
            random_scale=0.01,
            pseudocounts=True
        )
        assert config.match_threshold == 0.7
        assert config.global_factor == 0.5
        assert config.random_scale == 0.01
        assert config.pseudocounts is True

    def test_match_threshold_validation(self):
        """Test that match_threshold must be in [0, 1]."""
        # Valid values
        InitMSAConfig(match_threshold=0.0)
        InitMSAConfig(match_threshold=1.0)
        InitMSAConfig(match_threshold=0.5)

        # Invalid values
        with pytest.raises(
            ValidationError,
            match="match_threshold must be in the range"
        ):
            InitMSAConfig(match_threshold=-0.1)

        with pytest.raises(
            ValidationError,
            match="match_threshold must be in the range"
        ):
            InitMSAConfig(match_threshold=1.5)

    def test_global_factor_validation(self):
        """Test that global_factor must be in [0, 1]."""
        # Valid values
        InitMSAConfig(global_factor=0.0)
        InitMSAConfig(global_factor=1.0)
        InitMSAConfig(global_factor=0.5)

        # Invalid values
        with pytest.raises(
            ValidationError,
            match="global_factor must be in the range"
        ):
            InitMSAConfig(global_factor=-0.1)

        with pytest.raises(
            ValidationError,
            match="global_factor must be in the range"
        ):
            InitMSAConfig(global_factor=1.1)

    def test_random_scale_validation(self):
        """Test that random_scale must be positive."""
        # Valid values
        InitMSAConfig(random_scale=0.001)
        InitMSAConfig(random_scale=1.0)
        InitMSAConfig(random_scale=100.0)

        # Invalid values
        with pytest.raises(
            ValidationError,
            match="random_scale must be greater than 0"
        ):
            InitMSAConfig(random_scale=0)

        with pytest.raises(
            ValidationError,
            match="random_scale must be greater than 0"
        ):
            InitMSAConfig(random_scale=-0.001)

    def test_init_msa_serialization(self):
        """Test InitMSAConfig serialization to dict."""
        from pathlib import Path
        config = InitMSAConfig(
            from_msa=Path("/path/to/file.fasta"),
            match_threshold=0.6,
            global_factor=0.2
        )
        config_dict = config.model_dump()

        assert config_dict["from_msa"] == Path("/path/to/file.fasta")
        assert config_dict["match_threshold"] == 0.6
        assert config_dict["global_factor"] == 0.2
        assert config_dict["random_scale"] == 1e-3
        assert config_dict["pseudocounts"] is False

    def test_init_msa_deserialization(self):
        """Test InitMSAConfig deserialization from dict."""
        config_dict = {
            "from_msa": "/path/to/file.fasta",
            "match_threshold": 0.8,
            "global_factor": 0.3,
            "random_scale": 0.05,
            "pseudocounts": True
        }
        config = InitMSAConfig(**config_dict)

        assert str(config.from_msa) == "/path/to/file.fasta"
        assert config.match_threshold == 0.8
        assert config.global_factor == 0.3
        assert config.random_scale == 0.05
        assert config.pseudocounts is True

    def test_init_msa_in_configuration(self):
        """Test InitMSAConfig as part of Configuration."""
        from pathlib import Path
        config = Configuration(
            init_msa=InitMSAConfig(
                from_msa=Path("/data/msa.fasta"),
                match_threshold=0.75
            )
        )

        assert config.init_msa.from_msa == Path("/data/msa.fasta")
        assert config.init_msa.match_threshold == 0.75


class TestVisualizationConfig:
    """Tests for VisualizationConfig."""

    def test_visualization_config_defaults(self):
        """Test VisualizationConfig default values."""
        config = VisualizationConfig()
        assert config.logo == ""
        assert config.logo_gif == ""

    def test_visualization_config_custom_values(self):
        """Test VisualizationConfig with custom values."""
        config = VisualizationConfig(
            logo="output/logo.pdf",
            logo_gif="output/logo_animation.gif"
        )
        assert config.logo == "output/logo.pdf"
        assert config.logo_gif == "output/logo_animation.gif"

    def test_visualization_config_logo_only(self):
        """Test VisualizationConfig with only logo set."""
        config = VisualizationConfig(logo="results/sequence_logo.pdf")
        assert config.logo == "results/sequence_logo.pdf"
        assert config.logo_gif == ""

    def test_visualization_config_logo_gif_only(self):
        """Test VisualizationConfig with only logo_gif set."""
        config = VisualizationConfig(logo_gif="results/animation.gif")
        assert config.logo == ""
        assert config.logo_gif == "results/animation.gif"

    def test_visualization_serialization(self):
        """Test VisualizationConfig serialization to dict."""
        config = VisualizationConfig(
            logo="logo.pdf",
            logo_gif="logo.gif"
        )
        config_dict = config.model_dump()

        assert config_dict["logo"] == "logo.pdf"
        assert config_dict["logo_gif"] == "logo.gif"

    def test_visualization_deserialization(self):
        """Test VisualizationConfig deserialization from dict."""
        config_dict = {
            "logo": "path/to/logo.pdf",
            "logo_gif": "path/to/animation.gif"
        }
        config = VisualizationConfig(**config_dict)

        assert config.logo == "path/to/logo.pdf"
        assert config.logo_gif == "path/to/animation.gif"

    def test_visualization_in_configuration(self):
        """Test VisualizationConfig as part of Configuration."""
        config = Configuration(
            visualization=VisualizationConfig(
                logo="output/final_logo.pdf",
                logo_gif="output/training_animation.gif"
            )
        )

        assert config.visualization.logo == "output/final_logo.pdf"
        assert config.visualization.logo_gif == "output/training_animation.gif"

    def test_visualization_empty_strings_valid(self):
        """Test that empty strings are valid (default behavior)."""
        config = VisualizationConfig(logo="", logo_gif="")
        assert config.logo == ""
        assert config.logo_gif == ""


class TestLanguageModelConfig:
    """Tests for LanguageModelConfig."""

    def test_language_model_config_defaults(self):
        """Test LanguageModelConfig default values."""
        config = LanguageModelConfig()
        assert config.use_language_model is False
        assert config.plm_cache_dir is None
        assert config.language_model == "protT5"
        assert config.scoring_model_dim == 16
        assert config.scoring_model_activation == "sigmoid"
        assert config.scoring_model_suffix == ""
        assert config.temperature == 3.0
        assert config.temperature_mode == "trainable"
        assert config.use_L2 is False
        assert config.L2_match == 0.0
        assert config.L2_insert == 1000.0
        assert config.embedding_prior_components == 32

    def test_language_model_config_custom_values(self):
        """Test LanguageModelConfig with custom values."""
        config = LanguageModelConfig(
            use_language_model=True,
            plm_cache_dir="/path/to/cache",
            language_model="esm2",
            scoring_model_dim=32,
            scoring_model_activation="relu",
            scoring_model_suffix="_v2",
            temperature=5.0,
            temperature_mode="fixed",
            use_L2=True,
            L2_match=0.5,
            L2_insert=500.0,
            embedding_prior_components=64
        )
        assert config.use_language_model is True
        assert config.plm_cache_dir == "/path/to/cache"
        assert config.language_model == "esm2"
        assert config.scoring_model_dim == 32
        assert config.scoring_model_activation == "relu"
        assert config.scoring_model_suffix == "_v2"
        assert config.temperature == 5.0
        assert config.temperature_mode == "fixed"
        assert config.use_L2 is True
        assert config.L2_match == 0.5
        assert config.L2_insert == 500.0
        assert config.embedding_prior_components == 64

    def test_language_model_validation(self):
        """Test that language_model must be one of the allowed values."""
        # Valid values
        LanguageModelConfig(language_model="protT5")
        LanguageModelConfig(language_model="esm2")
        LanguageModelConfig(language_model="proteinBERT")

        # Invalid value
        with pytest.raises(
            ValidationError,
            match="language_model must be one of"
        ):
            LanguageModelConfig(language_model="invalid_model")

    def test_scoring_model_dim_validation(self):
        """Test that scoring_model_dim must be positive."""
        # Valid values
        LanguageModelConfig(scoring_model_dim=1)
        LanguageModelConfig(scoring_model_dim=100)

        # Invalid values
        with pytest.raises(
            ValidationError,
            match="scoring_model_dim must be greater than 0"
        ):
            LanguageModelConfig(scoring_model_dim=0)

        with pytest.raises(
            ValidationError,
            match="scoring_model_dim must be greater than 0"
        ):
            LanguageModelConfig(scoring_model_dim=-5)

    def test_embedding_prior_components_validation(self):
        """Test that embedding_prior_components must be positive."""
        # Valid values
        LanguageModelConfig(embedding_prior_components=1)
        LanguageModelConfig(embedding_prior_components=128)

        # Invalid values
        with pytest.raises(
            ValidationError,
            match="embedding_prior_components must be greater than 0"
        ):
            LanguageModelConfig(embedding_prior_components=0)

        with pytest.raises(
            ValidationError,
            match="embedding_prior_components must be greater than 0"
        ):
            LanguageModelConfig(embedding_prior_components=-10)

    def test_temperature_validation(self):
        """Test that temperature must be positive."""
        # Valid values
        LanguageModelConfig(temperature=0.1)
        LanguageModelConfig(temperature=10.0)

        # Invalid values
        with pytest.raises(
            ValidationError,
            match="temperature must be greater than 0"
        ):
            LanguageModelConfig(temperature=0)

        with pytest.raises(
            ValidationError,
            match="temperature must be greater than 0"
        ):
            LanguageModelConfig(temperature=-1.0)

    def test_l2_match_validation(self):
        """Test that L2_match must be non-negative."""
        # Valid values
        LanguageModelConfig(L2_match=0.0)
        LanguageModelConfig(L2_match=1.0)
        LanguageModelConfig(L2_match=100.0)

        # Invalid value
        with pytest.raises(
            ValidationError,
            match="L2_match must be non-negative"
        ):
            LanguageModelConfig(L2_match=-0.1)

    def test_l2_insert_validation(self):
        """Test that L2_insert must be non-negative."""
        # Valid values
        LanguageModelConfig(L2_insert=0.0)
        LanguageModelConfig(L2_insert=500.0)
        LanguageModelConfig(L2_insert=2000.0)

        # Invalid value
        with pytest.raises(
            ValidationError,
            match="L2_insert must be non-negative"
        ):
            LanguageModelConfig(L2_insert=-10.0)

    def test_language_model_serialization(self):
        """Test LanguageModelConfig serialization to dict."""
        config = LanguageModelConfig(
            use_language_model=True,
            language_model="proteinBERT",
            scoring_model_dim=24
        )
        config_dict = config.model_dump()

        assert config_dict["use_language_model"] is True
        assert config_dict["language_model"] == "proteinBERT"
        assert config_dict["scoring_model_dim"] == 24

    def test_language_model_deserialization(self):
        """Test LanguageModelConfig deserialization from dict."""
        config_dict = {
            "use_language_model": True,
            "plm_cache_dir": "/models/cache",
            "language_model": "esm2",
            "temperature": 2.5
        }
        config = LanguageModelConfig(**config_dict)

        assert config.use_language_model is True
        assert config.plm_cache_dir == "/models/cache"
        assert config.language_model == "esm2"
        assert config.temperature == 2.5

    def test_language_model_in_configuration(self):
        """Test LanguageModelConfig as part of Configuration."""
        config = Configuration(
            language_model=LanguageModelConfig(
                use_language_model=True,
                language_model="esm2",
                scoring_model_dim=48
            )
        )

        assert config.language_model.use_language_model is True
        assert config.language_model.language_model == "esm2"
        assert config.language_model.scoring_model_dim == 48

    def test_all_language_model_options(self):
        """Test that all three language model options work."""
        for model_name in ["protT5", "esm2", "proteinBERT"]:
            config = LanguageModelConfig(language_model=model_name)
            assert config.language_model == model_name


class TestAdvancedConfig:
    """Tests for AdvancedConfig."""

    def test_advanced_config_defaults(self):
        """Test AdvancedConfig default values."""
        config = AdvancedConfig()
        assert config.dist_out == ""
        assert config.alpha_flank == 7000
        assert config.alpha_single == 1e9
        assert config.alpha_global == 1e4
        assert config.alpha_flank_compl == 1
        assert config.alpha_single_compl == 1
        assert config.alpha_global_compl == 1
        assert config.inverse_gamma_alpha == 3.0
        assert config.inverse_gamma_beta == 0.5
        assert config.frozen_distances is False
        assert config.initial_distance == 0.05
        assert config.trainable_rate_matrices is False

    def test_advanced_config_custom_values(self):
        """Test AdvancedConfig with custom values."""
        config = AdvancedConfig(
            dist_out="distributions.txt",
            alpha_flank=5000,
            alpha_single=1e8,
            alpha_global=5000,
            alpha_flank_compl=2,
            alpha_single_compl=3,
            alpha_global_compl=4,
            inverse_gamma_alpha=2.0,
            inverse_gamma_beta=1.0,
            frozen_distances=True,
            initial_distance=0.1,
            trainable_rate_matrices=True
        )
        assert config.dist_out == "distributions.txt"
        assert config.alpha_flank == 5000
        assert config.alpha_single == 1e8
        assert config.alpha_global == 5000
        assert config.alpha_flank_compl == 2
        assert config.alpha_single_compl == 3
        assert config.alpha_global_compl == 4
        assert config.inverse_gamma_alpha == 2.0
        assert config.inverse_gamma_beta == 1.0
        assert config.frozen_distances is True
        assert config.initial_distance == 0.1
        assert config.trainable_rate_matrices is True

    def test_alpha_flank_validation(self):
        """Test that alpha_flank must be positive."""
        # Valid values
        AdvancedConfig(alpha_flank=0.1)
        AdvancedConfig(alpha_flank=10000)

        # Invalid values
        with pytest.raises(
            ValidationError,
            match="alpha_flank must be greater than 0"
        ):
            AdvancedConfig(alpha_flank=0)

        with pytest.raises(
            ValidationError,
            match="alpha_flank must be greater than 0"
        ):
            AdvancedConfig(alpha_flank=-100)

    def test_alpha_single_validation(self):
        """Test that alpha_single must be positive."""
        # Valid values
        AdvancedConfig(alpha_single=1)
        AdvancedConfig(alpha_single=1e10)

        # Invalid values
        with pytest.raises(
            ValidationError,
            match="alpha_single must be greater than 0"
        ):
            AdvancedConfig(alpha_single=0)

        with pytest.raises(
            ValidationError,
            match="alpha_single must be greater than 0"
        ):
            AdvancedConfig(alpha_single=-1)

    def test_alpha_global_validation(self):
        """Test that alpha_global must be positive."""
        # Valid values
        AdvancedConfig(alpha_global=100)
        AdvancedConfig(alpha_global=1e6)

        # Invalid values
        with pytest.raises(
            ValidationError,
            match="alpha_global must be greater than 0"
        ):
            AdvancedConfig(alpha_global=0)

        with pytest.raises(
            ValidationError,
            match="alpha_global must be greater than 0"
        ):
            AdvancedConfig(alpha_global=-500)

    def test_alpha_complement_validation(self):
        """Test that alpha complement parameters must be positive."""
        # Valid values
        AdvancedConfig(alpha_flank_compl=0.5)
        AdvancedConfig(alpha_single_compl=10)
        AdvancedConfig(alpha_global_compl=100)

        # Invalid values
        with pytest.raises(
            ValidationError,
            match="alpha_flank_compl must be greater than 0"
        ):
            AdvancedConfig(alpha_flank_compl=0)

        with pytest.raises(
            ValidationError,
            match="alpha_single_compl must be greater than 0"
        ):
            AdvancedConfig(alpha_single_compl=-1)

        with pytest.raises(
            ValidationError,
            match="alpha_global_compl must be greater than 0"
        ):
            AdvancedConfig(alpha_global_compl=-5)

    def test_inverse_gamma_alpha_validation(self):
        """Test that inverse_gamma_alpha must be positive."""
        # Valid values
        AdvancedConfig(inverse_gamma_alpha=0.1)
        AdvancedConfig(inverse_gamma_alpha=10.0)

        # Invalid values
        with pytest.raises(
            ValidationError,
            match="inverse_gamma_alpha must be greater than 0"
        ):
            AdvancedConfig(inverse_gamma_alpha=0)

        with pytest.raises(
            ValidationError,
            match="inverse_gamma_alpha must be greater than 0"
        ):
            AdvancedConfig(inverse_gamma_alpha=-2.0)

    def test_inverse_gamma_beta_validation(self):
        """Test that inverse_gamma_beta must be positive."""
        # Valid values
        AdvancedConfig(inverse_gamma_beta=0.01)
        AdvancedConfig(inverse_gamma_beta=5.0)

        # Invalid values
        with pytest.raises(
            ValidationError,
            match="inverse_gamma_beta must be greater than 0"
        ):
            AdvancedConfig(inverse_gamma_beta=0)

        with pytest.raises(
            ValidationError,
            match="inverse_gamma_beta must be greater than 0"
        ):
            AdvancedConfig(inverse_gamma_beta=-0.5)

    def test_initial_distance_validation(self):
        """Test that initial_distance must be positive."""
        # Valid values
        AdvancedConfig(initial_distance=0.001)
        AdvancedConfig(initial_distance=1.0)

        # Invalid values
        with pytest.raises(
            ValidationError,
            match="initial_distance must be greater than 0"
        ):
            AdvancedConfig(initial_distance=0)

        with pytest.raises(
            ValidationError,
            match="initial_distance must be greater than 0"
        ):
            AdvancedConfig(initial_distance=-0.1)

    def test_advanced_serialization(self):
        """Test AdvancedConfig serialization to dict."""
        config = AdvancedConfig(
            dist_out="output.txt",
            alpha_flank=8000,
            frozen_distances=True
        )
        config_dict = config.model_dump()

        assert config_dict["dist_out"] == "output.txt"
        assert config_dict["alpha_flank"] == 8000
        assert config_dict["frozen_distances"] is True

    def test_advanced_deserialization(self):
        """Test AdvancedConfig deserialization from dict."""
        config_dict = {
            "dist_out": "results.txt",
            "alpha_single": 5e8,
            "inverse_gamma_alpha": 4.0,
            "trainable_rate_matrices": True
        }
        config = AdvancedConfig(**config_dict)

        assert config.dist_out == "results.txt"
        assert config.alpha_single == 5e8
        assert config.inverse_gamma_alpha == 4.0
        assert config.trainable_rate_matrices is True

    def test_advanced_in_configuration(self):
        """Test AdvancedConfig as part of Configuration."""
        config = Configuration(
            advanced=AdvancedConfig(
                alpha_flank=6000,
                alpha_global=2e4,
                frozen_distances=True
            )
        )

        assert config.advanced.alpha_flank == 6000
        assert config.advanced.alpha_global == 2e4
        assert config.advanced.frozen_distances is True

    def test_all_alpha_parameters_positive(self):
        """Test that all alpha parameters accept positive values."""
        config = AdvancedConfig(
            alpha_flank=1.0,
            alpha_single=1.0,
            alpha_global=1.0,
            alpha_flank_compl=1.0,
            alpha_single_compl=1.0,
            alpha_global_compl=1.0
        )
        assert config.alpha_flank == 1.0
        assert config.alpha_single == 1.0
        assert config.alpha_global == 1.0
        assert config.alpha_flank_compl == 1.0
        assert config.alpha_single_compl == 1.0
        assert config.alpha_global_compl == 1.0


class TestInputOutputConfig:
    """Tests for InputOutputConfig."""

    def test_default_input_output_config(self):
        """Test that InputOutputConfig can be created with defaults."""
        config = InputOutputConfig()
        assert config.input_file == Path()
        assert config.output_file == Path()
        assert config.format == "a2m"
        assert config.input_format == "fasta"
        assert config.save_model == ""
        assert config.load_model == ""
        assert config.silent is False
        assert config.cuda_visible_devices == "default"
        assert config.work_dir == "tmp"
        assert config.convert is False

    def test_input_output_config_with_files(self):
        """Test InputOutputConfig with file paths."""
        config = InputOutputConfig(
            input_file=Path("input.fasta"),
            output_file=Path("output.a2m")
        )
        assert config.input_file == Path("input.fasta")
        assert config.output_file == Path("output.a2m")

    def test_input_output_config_with_string_paths(self):
        """Test InputOutputConfig accepts string paths."""
        config = InputOutputConfig(
            input_file=Path("sequences.fasta"),
            output_file=Path("alignment.a2m")
        )
        assert config.input_file == Path("sequences.fasta")
        assert config.output_file == Path("alignment.a2m")

    def test_format_validation_valid_formats(self):
        """Test format validation accepts valid formats."""
        valid_formats = ["a2m", "fasta", "stockholm", "clustal", "phylip"]
        for fmt in valid_formats:
            config = InputOutputConfig(format=fmt)
            assert config.format == fmt

    def test_format_validation_invalid_format(self):
        """Test format validation rejects invalid formats."""
        with pytest.raises(ValidationError, match="format must be one of"):
            InputOutputConfig(format="invalid_format")

    def test_input_format_validation_valid_formats(self):
        """Test input_format validation accepts valid formats."""
        valid_formats = ["fasta", "a2m", "stockholm", "clustal"]
        for fmt in valid_formats:
            config = InputOutputConfig(input_format=fmt)
            assert config.input_format == fmt

    def test_input_format_validation_invalid_format(self):
        """Test input_format validation rejects invalid formats."""
        with pytest.raises(ValidationError, match="input_format must be one of"):
            InputOutputConfig(input_format="invalid_format")

    def test_save_model_path(self):
        """Test save_model can be set to a file path."""
        config = InputOutputConfig(save_model="/path/to/model.pkl")
        assert config.save_model == "/path/to/model.pkl"

    def test_load_model_path(self):
        """Test load_model can be set to a file path."""
        config = InputOutputConfig(load_model="/path/to/model.pkl")
        assert config.load_model == "/path/to/model.pkl"

    def test_silent_flag(self):
        """Test silent flag can be set."""
        config = InputOutputConfig(silent=True)
        assert config.silent is True

    def test_cuda_visible_devices_default(self):
        """Test cuda_visible_devices with default value."""
        config = InputOutputConfig(cuda_visible_devices="default")
        assert config.cuda_visible_devices == "default"

    def test_cuda_visible_devices_cpu(self):
        """Test cuda_visible_devices can be set to CPU."""
        config = InputOutputConfig(cuda_visible_devices="-1")
        assert config.cuda_visible_devices == "-1"

    def test_cuda_visible_devices_single_gpu(self):
        """Test cuda_visible_devices with single GPU."""
        config = InputOutputConfig(cuda_visible_devices="0")
        assert config.cuda_visible_devices == "0"

    def test_cuda_visible_devices_multiple_gpus(self):
        """Test cuda_visible_devices with multiple GPUs."""
        config = InputOutputConfig(cuda_visible_devices="0,1,2")
        assert config.cuda_visible_devices == "0,1,2"

    def test_cuda_visible_devices_invalid_negative(self):
        """Test cuda_visible_devices rejects invalid negative values."""
        with pytest.raises(ValidationError, match="Device IDs must be non-negative"):
            InputOutputConfig(cuda_visible_devices="0,-2")

    def test_cuda_visible_devices_invalid_format(self):
        """Test cuda_visible_devices rejects non-numeric values."""
        with pytest.raises(ValidationError, match="cuda_visible_devices must be"):
            InputOutputConfig(cuda_visible_devices="abc")

    def test_work_dir_custom(self):
        """Test work_dir can be set to custom directory."""
        config = InputOutputConfig(work_dir="/custom/work/dir")
        assert config.work_dir == "/custom/work/dir"

    def test_convert_flag_true(self):
        """Test convert flag can be set to True."""
        config = InputOutputConfig(convert=True)
        assert config.convert is True

    def test_input_output_config_comprehensive(self):
        """Test InputOutputConfig with all parameters set."""
        config = InputOutputConfig(
            input_file=Path("input.fasta"),
            output_file=Path("output.stockholm"),
            format="stockholm",
            input_format="a2m",
            save_model="model.pkl",
            load_model="pretrained.pkl",
            silent=True,
            cuda_visible_devices="0,1",
            work_dir="/tmp/learnmsa",
            convert=True
        )
        assert config.input_file == Path("input.fasta")
        assert config.output_file == Path("output.stockholm")
        assert config.format == "stockholm"
        assert config.input_format == "a2m"
        assert config.save_model == "model.pkl"
        assert config.load_model == "pretrained.pkl"
        assert config.silent is True
        assert config.cuda_visible_devices == "0,1"
        assert config.work_dir == "/tmp/learnmsa"
        assert config.convert is True

    def test_input_output_in_configuration(self):
        """Test InputOutputConfig as part of Configuration."""
        config = Configuration(
            input_output=InputOutputConfig(
                input_file=Path("test.fasta"),
                output_file=Path("test.a2m"),
                format="fasta"
            )
        )

        assert config.input_output is not None
        assert config.input_output.input_file == Path("test.fasta")
        assert config.input_output.output_file == Path("test.a2m")
        assert config.input_output.format == "fasta"

    def test_input_output_serialization(self):
        """Test that InputOutputConfig can be serialized."""
        config = InputOutputConfig(
            input_file=Path("input.fasta"),
            output_file=Path("output.a2m"),
            save_model="model.pkl",
            convert=True
        )

        config_dict = config.model_dump()

        assert "input_file" in config_dict
        assert "output_file" in config_dict
        assert "format" in config_dict
        assert "save_model" in config_dict
        assert "convert" in config_dict
        assert config_dict["save_model"] == "model.pkl"
        assert config_dict["convert"] is True

