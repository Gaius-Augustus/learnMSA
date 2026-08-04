import tempfile

from learnMSA.util import clustering

# Sequence weighting shells out to mmseqs2. It is not a Python dependency, so
# it has to be installed alongside every environment learnMSA is tested in; a
# missing binary is an environment defect, not a reason to skip.


def test_clustering_with_ids() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        sequence_weights = clustering.compute_sequence_weights(
            "tests/data/failing_ids.fasta",
            temp_dir,
        )
