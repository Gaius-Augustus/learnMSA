import importlib.resources as resources
import logging
from typing import Sequence, cast

import numpy as np
import tensorflow as tf
from hidten.hmm import HMMConfig as HidtenHMMConfig
from hidten.tf.prior.dirichlet import TFDirichletPrior
from hidten.tf.prior.multivariate_normal import TFMVNormalPrior


def make_model(dim: int, layer : tf.keras.layers.Layer) -> tf.keras.Model:
    """Utility function that constructs a keras model over a layer for
    serialization.
    """
    input = tf.keras.Input(
        shape=(None, dim,), dtype=layer.dtype
    )
    loglik = layer(input)
    model = tf.keras.Model(inputs=[input], outputs=[loglik])
    return model

def make_dirichlet_model(
    initializer : np.ndarray | None = None,
    dim: int | None = None,
    components: int = 1,
    states: Sequence[int] = [1],
) -> tf.keras.Model:
    """Create a keras model with a TFDirichletPrior layer for the amino acid
    prior. If an initializer is provided, it is used to initialize the prior
    distribution.

    For multi-component priors, ``dim`` must be provided explicitly since
    the initializer length encodes both components and categories.
    """
    assert initializer is not None or dim is not None,\
        "Either initializer or dim must be provided."
    if dim is not None:
        n_dim = int(dim)
    elif components == 1:
        assert initializer is not None
        n_dim = int(initializer.shape[0])
    else:
        raise ValueError(
            "dim must be provided for multi-component Dirichlet priors."
        )
    prior = TFDirichletPrior(components=components)
    prior.hmm_config = HidtenHMMConfig(states=states)

    n_param = components * n_dim + components if components > 1 else n_dim

    # Configure parameters
    if initializer is not None:
        prior.initializer = initializer
    else:
        prior.initializer = np.ones((n_param,))

    # Share concentrations across all states
    prior.share = np.tile(np.arange(n_param), reps=sum(states))

    prior.build((None, None, None, n_dim))
    return make_model(n_dim, prior)

def load_dirichlet(
    name: str, dim: int, components: int = 1, states: Sequence[int] = [1]
) -> TFDirichletPrior:
    """Load weights from a resource file into the given model.

    Args:
        name (str): The name of the weight resource file (without extension).
        dim (int): The dimension of the Dirichlet prior.
        components (int): The number of mixture components.
        states (Sequence[int]): The number of states in each head.
    """
    model = make_dirichlet_model(
        dim = dim, components = components, states = states
    )
    resource = resources.files("learnMSA.hmm.weights") / f"{name}.h5"
    model.load_weights(str(resource))
    prior = cast(TFDirichletPrior, model.layers[1])
    _warn_if_degenerate(prior, name)
    return prior


#: A concentration below this is treated as a collapsed dimension.
DEGENERATE_ALPHA: float = 1e-3

#: ...but only within a component that is otherwise informative.
INFORMATIVE_ALPHA: float = 1.0


def _warn_if_degenerate(prior: TFDirichletPrior, name: str) -> None:
    """Warn about components with a collapsed dimension.

    A concentration below one makes the density diverge at the simplex
    boundary, which is fine on its own: the flat 3Di priors consist entirely
    of such values. What is not fine is a concentration of ~0 sitting *inside*
    an otherwise informative component. It asserts that this letter is never
    emitted, which is a fitting artifact rather than a modelled belief, and it
    is what the prior behind the reported NaN run looked like.

    Judged per component, so a dead component of a mixture -- all
    concentrations tiny, and carrying a near-zero mixture weight -- does not
    trigger the warning.
    """
    matrix = prior.matrix()
    if prior.config.components > 1:
        # Drop the trailing mixture coefficients.
        matrix = prior._slice_concentrations(matrix)
    alpha = matrix.numpy().reshape(-1, prior.input_dim)
    # Padding states carry all-zero rows and are not concentrations.
    alpha = alpha[alpha.sum(axis=-1) > 0]
    if alpha.size == 0:
        return
    degenerate = (
        (alpha.min(axis=-1) < DEGENERATE_ALPHA)
        & (alpha.max(axis=-1) > INFORMATIVE_ALPHA)
    )
    if not degenerate.any():
        return
    bad = alpha[degenerate]
    logging.warning(
        "Dirichlet prior '%s' has %d component(s) with a collapsed "
        "dimension: a concentration of %.3g next to a maximum of %.4g. That "
        "letter is effectively forbidden, which is usually a fitting "
        "artifact; consider refitting the prior.",
        name, int(degenerate.sum()), float(bad.min()), float(bad.max()),
    )

def make_mvn_model(
    dim: int,
    initializer : np.ndarray | None = None,
    components: int = 1,
    states: Sequence[int] = [1],
) -> tf.keras.Model:
    """Create a keras model with a TFMVNormalPrior layer for the multivariate
    normal prior. If an initializer is provided, it is used to initialize the
    prior distribution.

    Args:
        initializer: Optional initial parameter values.
        dim: The dimension of the observations (means + variances, so 2*D).
        components: The number of mixture components.
        states: The number of states in each head.
    """
    prior = TFMVNormalPrior(components=components)
    prior.hmm_config = HidtenHMMConfig(states=states)

    n_param = components * 2 * dim + components if components > 1 else 2 * dim

    # Configure parameters
    if initializer is not None:
        prior.initializer = initializer
    else:
        prior.initializer = np.zeros((n_param,))

    # Share parameters across all states
    prior.share = np.tile(np.arange(n_param), reps=sum(states))

    prior.build((None, None, 2 * dim))
    return make_model(dim, prior)

def load_mvn(
    name: str, dim: int, components: int = 1, states: Sequence[int] = [1]
) -> TFMVNormalPrior:
    """Load weights from a resource file into the given model.

    Args:
        name (str): The name of the weight resource file (without extension).
        dim (int): The dimension of the multivariate normal prior.
        components (int): The number of mixture components.
        states (Sequence[int]): The number of states in each head.
    """
    model = make_mvn_model(dim=dim, components=components, states=states)
    resource = resources.files("learnMSA.hmm.weights") / f"{name}.h5"
    model.load_weights(str(resource))
    return cast(TFMVNormalPrior, model.layers[1])
