import tempfile
from learnMSA.msa_hmm import Align


def test_clustering_with_ids() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        sequence_weights = Align.compute_sequence_weights(
            "tests/data/failing_ids.fasta",
            temp_dir,
        )
