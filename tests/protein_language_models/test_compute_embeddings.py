import pytest

from learnMSA.config.config import Configuration
from learnMSA.model.context import LearnMSAContext
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
    config = Configuration()
    config.language_model.language_model = "protT5"
    config.language_model.scoring_model_dim = 32
    context = LearnMSAContext(config, dataset)
    embeddings = compute_embeddings(dataset, context)
    assert embeddings.cache.shape == (4 + 3 + 5, 32)
