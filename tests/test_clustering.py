import tempfile
from learnMSA.msa_hmm import clustering


def test_clustering_with_ids() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        sequence_weights = clustering.compute_sequence_weights(
            "tests/data/failing_ids.fasta",
            temp_dir,
        )
