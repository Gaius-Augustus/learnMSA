import numpy as np
from learnMSA.msa_hmm.SequenceDataset import AlignedDataset
from learnMSA.protein_language_models import DataPipeline


def test_column_occupancies() -> None:
    fasta = AlignedDataset("tests/data/felix_msa.fa")
    column_occupancies = DataPipeline._get_column_occupancies(fasta)
    np.testing.assert_almost_equal(
        column_occupancies, [1./3, 2./3, 2./3, 1, 1, 2./3, 2./3, 1.]
    )
