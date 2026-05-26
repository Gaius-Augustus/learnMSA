import numpy as np
import tensorflow as tf
import scipy

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
        if isinstance(self.value, np.ndarray):
            value = self.value.tolist()
        else:
            value = self.value
        return {"value": value}

    @classmethod
    def from_config(cls, config):
        return cls(np.array(config["value"]))


def make_default_anc_probs_init(
    num_models: int,
    num_components: int = 1,
    shared_equilibrium: bool = True,
    shared_exchangeabilities: bool = True,
    exchangeability_noise_std: float = 0.05,
    equilibrium_noise_std: float = 0.01,
    seed: int | None = None,
) -> tuple[ConstantInitializer, ConstantInitializer]:

    rng = np.random.default_rng(seed)
    R, p = LG(SequenceDataset._default_alphabet[:20])  # (D, D), (D,)
    D = R.shape[0]

    # Build exchangeability initializer: (H, 1, K_R, D, D)
    if shared_exchangeabilities:
        R_init = inverse_softplus(R + 1e-32).numpy()
        exchangeability_stack = np.tile(
            R_init[None, None, None], [num_models, 1, 1, 1, 1]
        )  # (H, 1, 1, D, D)
    else:
        eps = rng.normal(
            0.0, exchangeability_noise_std, (num_models, num_components, D, D)
        )
        eps = 0.5 * (eps + np.transpose(eps, (0, 1, 3, 2)))  # symmetrize
        perturbed_R = R * np.exp(eps)  # (H, K, D, D) via broadcasting
        diag_idx = np.arange(D)
        perturbed_R[:, :, diag_idx, diag_idx] = R[diag_idx, diag_idx]
        exchangeability_stack = inverse_softplus(
            perturbed_R + 1e-32
        ).numpy()[:, None]  # (H, 1, K, D, D)

    # Build equilibrium initializer: (H, 1, K_p, D)
    log_p = np.log(p)
    if shared_equilibrium:
        # (H, 1, 1, D)
        log_p_stack = np.tile(log_p[None, None, None], [num_models, 1, 1, 1])
    else:
        eps = rng.normal(
            0.0, equilibrium_noise_std, (num_models, num_components, D)
        )
        perturbed_log_p = log_p + eps  # (H, K, D) via broadcasting
        perturbed_log_p -= scipy.special.logsumexp(
            perturbed_log_p, axis=-1, keepdims=True
        )
        log_p_stack = perturbed_log_p[:, None]  # (H, 1, K, D)

    return (
        ConstantInitializer(exchangeability_stack),
        ConstantInitializer(log_p_stack),
    )
