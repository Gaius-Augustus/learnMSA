import pytest

from learnMSA.config.hmm import PHMMConfig
from learnMSA.config.structure import StructureConfig
from learnMSA.hmm.tf.layer import PHMMLayer


@pytest.fixture
def phmm_config() -> PHMMConfig:
    config = PHMMConfig()
    config.match_emissions = [
        [ # head 1
            [1] + [0] * 22,
            [0] + [1] + [0] * 21,
            [0] * 2 + [1] + [0] * 20,
            [0] * 3 + [1] + [0] * 19
        ],
        [ # head 2
            [0] * 4 + [1] + [0] * 18,
            [0] * 5 + [1] + [0] * 17,
            [0] * 6 + [1] + [0] * 16
        ]
    ]
    return config

@pytest.fixture
def structural_config() -> StructureConfig:
    config = StructureConfig()
    config.use_structure = True
    config.match_emissions = [
        [ # head 1
            [1] + [0] * 19,
            [0] + [1] + [0] * 18,
            [0] * 2 + [1] + [0] * 17,
            [0] * 3 + [1] + [0] * 16
        ],
        [ # head 2
            [0] * 4 + [1] + [0] * 15,
            [0] * 5 + [1] + [0] * 14,
            [0] * 6 + [1] + [0] * 13
        ]
    ]
    return config

def test_phmm_multi_emitter(
    phmm_config: PHMMConfig, structural_config: StructureConfig
) -> None:
    lengths = [4, 3]
    layer = PHMMLayer(
        lengths=lengths,
        config=phmm_config,
        structural_config=structural_config
    )
    layer.build(input_shape=(
        (None, None, None, 23),
        (None, None, None, 20),
        (None, None, None, 1)
    ))
    print(layer.hmm.emitter[0].matrix()[0, :4])
    print(layer.hmm.emitter[1].matrix()[0, :4])
    print(layer.hmm.emitter[0].matrix()[1, :4])
    print(layer.hmm.emitter[1].matrix()[1, :4])
