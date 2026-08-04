import torch


def inverse_softplus(features: torch.Tensor) -> torch.Tensor:
    """The inverse of ``softplus``, i.e. ``log(exp(x) - 1)``.

    Computed in float64 to prevent overflow of large entries and cast back,
    matching :func:`learnMSA.tree.tf.util.inverse_softplus` element for
    element.
    """
    epsilon = 1e-16
    features64 = features.double()
    result = torch.log(torch.expm1(features64) + epsilon)
    return result.to(features.dtype)
