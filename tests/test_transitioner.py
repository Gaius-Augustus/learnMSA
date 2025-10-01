from pydantic import ValidationError

from learnMSA.tf import PHMMExplicitTransitioner
from learnMSA import ProfileHMMConfig


def test_matrix():
    lengths = [4]
    transitioner = PHMMExplicitTransitioner(lengths=lengths)
    transitioner.hmm_config = ProfileHMMConfig(lengths=lengths)
    print(transitioner.matrix())