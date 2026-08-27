import pytest

from learnMSA.config import LanguageModelConfig
from learnMSA.protein_language_models.compute_embeddings import \
    compute_embeddings
from learnMSA.util.sequence_dataset import SequenceDataset


@pytest.fixture
def dataset() -> SequenceDataset:
    return SequenceDataset(
        sequences=[
            ("seq1", "ACGT"),
            ("seq2", "ACG"),
            ("seq3", "ACGTA"),
        ]
    )

@pytest.mark.filterwarnings(
    "ignore:builtin type SwigPyPacked has no __module__ attribute:DeprecationWarning",
    "ignore:builtin type SwigPyObject has no __module__ attribute:DeprecationWarning",
)
def test_compute_embeddings(dataset: SequenceDataset) -> None:
    config = LanguageModelConfig()
    config.language_model = "zeros" # for quick testing
    config.scoring_model_dim = 32
    embeddings = compute_embeddings(dataset, config)
    assert embeddings.cache.shape == (4 + 3 + 5, 32)
