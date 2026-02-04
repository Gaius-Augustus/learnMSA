import numpy as np
import tensorflow as tf

from learnMSA.tree.util import LG_paml, inverse_softplus, parse_paml
from learnMSA.util.sequence_dataset import SequenceDataset


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


R, p = parse_paml(LG_paml, SequenceDataset._default_alphabet[:-1])
exchangeability_init = inverse_softplus(R + 1e-32).numpy()


def make_default_anc_probs_init(num_models):
    exchangeability_stack = np.stack([exchangeability_init]*num_models, axis=0)
    log_p_stack = np.stack([np.log(p)]*num_models, axis=0)
    exchangeability_stack = np.expand_dims(exchangeability_stack, axis=1) #"k" in AncProbLayer
    log_p_stack = np.expand_dims(log_p_stack, axis=1) #"k" in AncProbLayer
    return [ConstantInitializer(-3),
            ConstantInitializer(exchangeability_stack),
            ConstantInitializer(log_p_stack)]
