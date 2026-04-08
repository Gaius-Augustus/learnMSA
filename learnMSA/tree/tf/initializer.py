import numpy as np
import tensorflow as tf

from learnMSA.tree.tf.util import inverse_softplus
from learnMSA.util.sequence_dataset import SequenceDataset

from evoten.substitution_models import LG

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
        return {"value": self.value.tolist() if isinstance(self.value, np.ndarray) else self.value}

    @classmethod
    def from_config(cls, config):
        return cls(np.array(config["value"]))


def make_default_anc_probs_init(
    num_models: int
) -> list[tf.keras.initializers.Initializer]:
    R, p = LG(SequenceDataset._default_alphabet[:20])# (D, D), (D,)
    exchangeability_init = inverse_softplus(R + 1e-32).numpy()

    # Stack to (H, D, D), (H, D)
    exchangeability_stack = np.stack([exchangeability_init]*num_models, axis=0)
    log_p_stack = np.stack([np.log(p)]*num_models, axis=0)

    # Expand to (H, 1, D, D), (H, 1, D) for broadcasting in the layer
    exchangeability_stack = np.expand_dims(exchangeability_stack, axis=1)
    log_p_stack = np.expand_dims(log_p_stack, axis=1)

    return [ConstantInitializer(-3),
            ConstantInitializer(exchangeability_stack),
            ConstantInitializer(log_p_stack)]
