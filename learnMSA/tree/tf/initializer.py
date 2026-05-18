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
    num_models: int,
    num_components: int = 1,
    shared_equilibrium: bool = True,
    exchangeability_noise_std: float = 0.05,
    equilibrium_noise_std: float = 0.01,
    seed: int | None = None,
) -> list[tf.keras.initializers.Initializer]:

    rng = np.random.default_rng(seed)

    # LG exchangeabilities and equilibrium frequencies
    R, p = LG(SequenceDataset._default_alphabet[:20])  # (D, D), (D,)

    # Parameterization space
    # Exchangeabilities are assumed positive and parameterized through softplus
    exchangeability_init = inverse_softplus(R + 1e-32).numpy()

    # Log equilibrium frequencies
    log_p_init = np.log(p)

    D = R.shape[0]

    # ------------------------------------------------------------------
    # Build component-specific perturbations
    # ------------------------------------------------------------------

    exchangeability_models = []
    equilibrium_models = []

    for h in range(num_models):

        exchangeability_components = []
        equilibrium_components = []

        for k in range(num_components):

            # ----------------------------------------------------------
            # Exchangeability perturbation
            # multiplicative perturbation in original space:
            #
            #   R_k = R * exp(eps)
            #
            # implemented in unconstrained parameter space
            # ----------------------------------------------------------

            eps_R = rng.normal(
                loc=0.0,
                scale=exchangeability_noise_std,
                size=(D, D),
            )

            # Keep symmetric structure
            eps_R = 0.5 * (eps_R + eps_R.T)

            perturbed_R = R * np.exp(eps_R)

            # Optional: keep diagonal untouched
            np.fill_diagonal(perturbed_R, np.diag(R))

            perturbed_exchangeability = inverse_softplus(
                perturbed_R + 1e-32
            ).numpy()

            exchangeability_components.append(
                perturbed_exchangeability
            )

            # ----------------------------------------------------------
            # Equilibrium perturbation
            # small perturbation in log-probability simplex coordinates
            # ----------------------------------------------------------

            if shared_equilibrium:
                continue

            eps_p = rng.normal(
                loc=0.0,
                scale=equilibrium_noise_std,
                size=(D,),
            )

            perturbed_log_p = log_p_init + eps_p

            # renormalize
            perturbed_log_p = (
                perturbed_log_p
                - scipy.special.logsumexp(perturbed_log_p)
            )

            equilibrium_components.append(perturbed_log_p)

        exchangeability_components = np.stack(
            exchangeability_components,
            axis=0,
        )  # (K, D, D)

        exchangeability_models.append(
            exchangeability_components
        )

        if not shared_equilibrium:
            equilibrium_components = np.stack(
                equilibrium_components,
                axis=0,
            )  # (K, D)

            equilibrium_models.append(
                equilibrium_components
            )

    # ------------------------------------------------------------------
    # Stack models
    # ------------------------------------------------------------------

    exchangeability_stack = np.stack(
        exchangeability_models,
        axis=0,
    )  # (H, K, D, D)

    # Add I=1 dimension
    exchangeability_stack = np.expand_dims(
        exchangeability_stack,
        axis=1,
    )  # (H, 1, K, D, D)

    if shared_equilibrium:

        log_p_stack = np.stack(
            [log_p_init] * num_models,
            axis=0,
        )  # (H, D)

        log_p_stack = np.expand_dims(
            np.expand_dims(log_p_stack, axis=1),
            axis=2,
        )  # (H, 1, 1, D)

    else:

        log_p_stack = np.stack(
            equilibrium_models,
            axis=0,
        )  # (H, K, D)

        log_p_stack = np.expand_dims(
            log_p_stack,
            axis=1,
        )  # (H, 1, K, D)

    # ------------------------------------------------------------------
    # Mixture logits
    # Slight random asymmetry around uniform
    # ------------------------------------------------------------------

    mixture_logits = rng.normal(
        loc=0.0,
        scale=0.1,
        size=(num_models, 1, num_components),
    )

    return [
        ConstantInitializer(mixture_logits),
        ConstantInitializer(exchangeability_stack),
        ConstantInitializer(log_p_stack),
    ]
