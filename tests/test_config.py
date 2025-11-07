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
        assert config.training.num_model == 4
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
        assert config.training.num_model == 3

    def test_num_model_setter(self):
        """Test that num_model can be set directly."""
        config = Configuration()
        config.training.num_model = 10
        assert config.training.num_model == 10

    def test_num_model_setter_validation(self):
        """Test that num_model validates value >= 1."""
        config = Configuration()
        with pytest.raises(
            ValueError, match="num_model must be greater than or equal to 1"
        ):
            config.training.num_model = 0
        with pytest.raises(
            ValueError, match="num_model must be greater than or equal to 1"
        ):
            config.training.num_model = -5

    def test_num_model_initialization_with_alias(self):
        """Test that num_model can be set during initialization."""
        training_config = TrainingConfig(num_model=7)
        assert training_config.num_model == 7
        config = Configuration(training=training_config)
        assert config.training.num_model == 7

    def test_num_model_length_init_precedence(self):
        """Test that length_init takes precedence over num_model."""
        config = Configuration(
            training=TrainingConfig(num_model=10, length_init=[5, 10])
        )
        # length_init has 2 elements, so num_model should be 2
        assert config.training.num_model == 2

        # Setting num_model shouldn't change the result while length_init is set
        config.training.num_model = 20
        assert config.training.num_model == 2  # Still returns len(length_init)

    def test_num_model_after_clearing_length_init(self):
        """Test num_model behavior when length_init is set then cleared."""
        training_config = TrainingConfig(num_model=5, length_init=[10, 20])
        assert training_config.num_model == 2  # From length_init

        # Clear length_init
        training_config.length_init = None
        # Should now use the _num_model value (which was set to 5 during init)
        assert training_config.num_model == 5

        # Set a different length_init
        training_config.length_init = [1, 2, 3]
        assert training_config.num_model == 3


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        assert config.batch_size == -1
        assert config.tokens_per_batch == -1
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
        config1 = TrainingConfig(auto_crop=True)
        assert config1.auto_crop

        config2 = TrainingConfig(auto_crop=False, crop=100)
        assert config2.crop == 100

        with pytest.raises(ValidationError):
            TrainingConfig(crop=0)  # Must be > 0


class TestConfigurationSerialization:
    """Tests for configuration serialization/deserialization."""

    def test_configuration_to_dict(self):
        """Test that Configuration can be serialized to dict."""
        config = Configuration(training=TrainingConfig(num_model=5))
        config_dict = config.model_dump()
        assert "training" in config_dict
        assert "init_msa" in config_dict
        assert "language_model" in config_dict
        assert "visualization" in config_dict
        assert "advanced" in config_dict
        # num_model is now in the training dict
        assert "num_model" in config_dict["training"]
        assert config_dict["training"]["num_model"] == 5

    def test_configuration_from_dict(self):
        """Test that Configuration can be created from dict."""
        config_dict = {
            "training": {"num_model": 7, "learning_rate": 0.05}
        }
        config = Configuration(**config_dict)  # type: ignore
        assert config.training.num_model == 7
        assert config.training.learning_rate == 0.05

    def test_configuration_roundtrip_with_num_model(self):
        """Test that num_model is preserved through serialization."""
        config = Configuration(
            training=TrainingConfig(num_model=10, learning_rate=0.01)
        )
        assert config.training.num_model == 10

        # Serialize and deserialize - num_model is now included in training dict
        config_dict = config.model_dump()
        assert config_dict["training"]["num_model"] == 10

        config2 = Configuration(**config_dict)
        assert config2.training.num_model == 10
        assert config2.training.learning_rate == 0.01

    def test_configuration_with_length_init_serialization(self):
        """Test serialization when length_init is set."""
        config = Configuration(
            training=TrainingConfig(length_init=[10, 20, 30])
        )
        config_dict = config.model_dump()
        # num_model is computed from length_init and included in serialization
        assert config.training.num_model == 3
        assert config_dict["training"]["num_model"] == 3
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
#         assert config.training.num_model == 5
#         assert config.training.epochs == [20, 20, 20]

#         # Modify training config to add length_init
#         config.training.length_init = [10, 20, 30]

#         # num_model should now reflect length_init
#         assert config.training.num_model == 3

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

    def test_init_msa_validation(self):
        """Test InitMSAConfig field validations."""
        # Valid match_threshold values
        for val in [0.0, 0.5, 1.0]:
            InitMSAConfig(match_threshold=val)
        
        # Invalid match_threshold values
        with pytest.raises(ValidationError, match="match_threshold must be in the range"):
            InitMSAConfig(match_threshold=-0.1)
        with pytest.raises(ValidationError, match="match_threshold must be in the range"):
            InitMSAConfig(match_threshold=1.5)

        # Valid global_factor values
        for val in [0.0, 0.5, 1.0]:
            InitMSAConfig(global_factor=val)
        
        # Invalid global_factor values
        with pytest.raises(ValidationError, match="global_factor must be in the range"):
            InitMSAConfig(global_factor=-0.1)
        with pytest.raises(ValidationError, match="global_factor must be in the range"):
            InitMSAConfig(global_factor=1.1)

        # Valid random_scale values
        for val in [0.001, 1.0, 100.0]:
            InitMSAConfig(random_scale=val)
        
        # Invalid random_scale values
        with pytest.raises(ValidationError, match="random_scale must be greater than 0"):
            InitMSAConfig(random_scale=0)
        with pytest.raises(ValidationError, match="random_scale must be greater than 0"):
            InitMSAConfig(random_scale=-0.001)

    def test_init_msa_serialization(self):
        """Test InitMSAConfig serialization and deserialization."""
        from pathlib import Path
        
        # Serialization
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

        # Deserialization
        config_dict2 = {
            "from_msa": "/path/to/file.fasta",
            "match_threshold": 0.8,
            "global_factor": 0.3,
            "random_scale": 0.05,
            "pseudocounts": True
        }
        config2 = InitMSAConfig(**config_dict2)

        assert str(config2.from_msa) == "/path/to/file.fasta"
        assert config2.match_threshold == 0.8
        assert config2.global_factor == 0.3
        assert config2.random_scale == 0.05
        assert config2.pseudocounts is True

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

    def test_visualization_config_values(self):
        """Test VisualizationConfig with various combinations."""
        # Both set
        config1 = VisualizationConfig(
            logo="output/logo.pdf",
            logo_gif="output/logo_animation.gif"
        )
        assert config1.logo == "output/logo.pdf"
        assert config1.logo_gif == "output/logo_animation.gif"

        # Only logo set
        config2 = VisualizationConfig(logo="results/sequence_logo.pdf")
        assert config2.logo == "results/sequence_logo.pdf"
        assert config2.logo_gif == ""

        # Only logo_gif set
        config3 = VisualizationConfig(logo_gif="results/animation.gif")
        assert config3.logo == ""
        assert config3.logo_gif == "results/animation.gif"

    def test_visualization_serialization(self):
        """Test VisualizationConfig serialization and deserialization."""
        # Serialization
        config = VisualizationConfig(logo="logo.pdf", logo_gif="logo.gif")
        config_dict = config.model_dump()
        assert config_dict["logo"] == "logo.pdf"
        assert config_dict["logo_gif"] == "logo.gif"

        # Deserialization
        config2 = VisualizationConfig(**{"logo": "path/to/logo.pdf", "logo_gif": "path/to/animation.gif"})
        assert config2.logo == "path/to/logo.pdf"
        assert config2.logo_gif == "path/to/animation.gif"

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

        # Valid values (must be positive)
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

        # Valid values (must be positive)
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
        }
        config = AdvancedConfig(**config_dict)

        assert config.dist_out == "results.txt"
        assert config.alpha_single == 5e8
        assert config.inverse_gamma_alpha == 4.0

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
        assert config.verbose is True
        assert config.cuda_visible_devices == "default"
        assert config.work_dir == "tmp"
        assert config.convert is False

    def test_input_output_config_with_files(self):
        """Test InputOutputConfig with file paths and string paths."""
        # With Path objects
        config1 = InputOutputConfig(
            input_file=Path("input.fasta"),
            output_file=Path("output.a2m")
        )
        assert config1.input_file == Path("input.fasta")
        assert config1.output_file == Path("output.a2m")

        # With string paths (converted to Path)
        config2 = InputOutputConfig(
            input_file=Path("sequences.fasta"),
            output_file=Path("alignment.a2m")
        )
        assert config2.input_file == Path("sequences.fasta")
        assert config2.output_file == Path("alignment.a2m")

    def test_format_validation(self):
        """Test format and input_format validation."""
        # Valid output formats
        for fmt in ["a2m", "fasta", "stockholm", "clustal", "phylip"]:
            config = InputOutputConfig(format=fmt)
            assert config.format == fmt

        # Invalid output format
        with pytest.raises(ValidationError, match="format must be one of"):
            InputOutputConfig(format="invalid_format")

        # Valid input formats
        for fmt in ["fasta", "a2m", "stockholm", "clustal"]:
            config = InputOutputConfig(input_format=fmt)
            assert config.input_format == fmt

        # Invalid input format
        with pytest.raises(ValidationError, match="input_format must be one of"):
            InputOutputConfig(input_format="invalid_format")

    def test_model_paths(self):
        """Test save_model and load_model paths."""
        config = InputOutputConfig(
            save_model="/path/to/save.pkl",
            load_model="/path/to/load.pkl"
        )
        assert config.save_model == "/path/to/save.pkl"
        assert config.load_model == "/path/to/load.pkl"

    def test_cuda_visible_devices_validation(self):
        """Test cuda_visible_devices with various valid and invalid values."""
        # Valid values
        for value in ["default", "-1", "0", "0,1,2"]:
            config = InputOutputConfig(cuda_visible_devices=value)
            assert config.cuda_visible_devices == value

        # Invalid: negative device IDs
        with pytest.raises(ValidationError, match="Device IDs must be non-negative"):
            InputOutputConfig(cuda_visible_devices="0,-2")

        # Invalid: non-numeric values
        with pytest.raises(ValidationError, match="cuda_visible_devices must be"):
            InputOutputConfig(cuda_visible_devices="abc")

    def test_input_output_config_comprehensive(self):
        """Test InputOutputConfig with all parameters set."""
        config = InputOutputConfig(
            input_file=Path("input.fasta"),
            output_file=Path("output.stockholm"),
            format="stockholm",
            input_format="a2m",
            save_model="model.pkl",
            load_model="pretrained.pkl",
            verbose=False,
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
        assert config.verbose is False
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

