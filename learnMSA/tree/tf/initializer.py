import numpy as np
import tensorflow as tf

from learnMSA.tree.tf.util import inverse_softplus
from learnMSA.util.sequence_dataset import SequenceDataset

from evoten.substitution_models import LG, foldseek_3Di, AF_3Di

class ConstantInitializer(tf.keras.initializers.Constant):

    def __init__(self, value):
        super(ConstantInitializer, self).__init__(value)

    def __repr__(self):
        if np.isscalar(self.value):
            return f"Const({self.value})"
        elif isinstance(self.value, list):
            return f"Const(size={len(self.value)})"
        else:
            return f"Const(shape={self.value.shape})"

    def get_config(self):  # To support serialization
        if isinstance(self.value, np.ndarray):
            value = self.value.tolist()
        else:
            value = self.value
        return {"value": value}

    @classmethod
    def from_config(cls, config):
        return cls(np.array(config["value"]))


def make_substitution_model_init(
    num_models: int,
    type: str = "LG",
    num_components: int = 1,
    shared_equilibrium: bool = True,
    shared_exchangeabilities: bool = True,
    alphabet: str = SequenceDataset._default_alphabet[:20],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constructs initializers for exchangeabilities and equilibrium frequencies
    based on an existing substitution model.
    """

    # (D, D), (D,)
    if type == "LG":
        R, p = LG(alphabet)
    elif type == "foldseek_3Di":
        R, p = foldseek_3Di(alphabet)
    elif type == "AF_3Di":
        R, p = AF_3Di(alphabet)
    else:
        raise ValueError(f"Unknown substitution model type: {type}")

    # Build exchangeability initializer: (H, 1, K_R, D, D)
    R_init = inverse_softplus(R + 1e-32).numpy()
    exchangeability_stack = np.tile(
        R_init[None, None, None], [num_models, 1, 1, 1, 1]
    )  # (H, 1, 1, D, D)
    if not shared_exchangeabilities and num_components > 1:
        exchangeability_stack = np.tile(
            exchangeability_stack, [1, 1, num_components, 1, 1]
        )  # (H, 1, K, D, D)

    # Build equilibrium initializer: (H, 1, K_p, D)
    log_p_stack = np.tile(
        np.log(p)[None, None, None], [num_models, 1, 1, 1]
    )  # (H, 1, 1, D)
    if not shared_equilibrium and num_components > 1:
        log_p_stack = np.tile(
            log_p_stack, [1, 1, num_components, 1]
        )  # (H, 1, K, D)

    return exchangeability_stack, log_p_stack
