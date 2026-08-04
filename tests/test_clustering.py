import shutil
import tempfile

import pytest

from learnMSA.util import clustering

# Sequence weighting shells out to mmseqs2, which is not a Python dependency
# and so is not guaranteed to be present in every environment learnMSA is
# tested in.
requires_mmseqs = pytest.mark.skipif(
    shutil.which("mmseqs") is None,
    reason="mmseqs2 is not installed",
)


@requires_mmseqs
def test_clustering_with_ids() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        sequence_weights = clustering.compute_sequence_weights(
            "tests/data/failing_ids.fasta",
            temp_dir,
        )
