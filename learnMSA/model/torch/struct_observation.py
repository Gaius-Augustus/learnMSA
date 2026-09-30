"""Observation model for misclassified structural tokens.

Structural tokens such as 3Di are often predicted from sequence (e.g. by
ProstT5) rather than computed from a structure, so the observed token o can
differ from the true token y of the residue. This layer models that
misclassification separately from evolution: the one-hot observation is
replaced by the likelihood vector

    v_y = M[y, o] = P(observed o | true y)

over the residue's true token. The ancestral-probabilities layer and the
emitters then act on v exactly as they act on a one-hot input, because both
are linear in it. ``M = I`` reproduces error-free observations.
"""

import numpy as np
import torch

from learnMSA.config.structure import StructureConfig
from learnMSA.hmm.priors import prior_path

#: Strength bounds used when the mixing weight is trainable, so that its
#: logit stays finite.
_MIN_STRENGTH = 1e-4


def load_confusion(name: str) -> np.ndarray:
    """Load a shipped row-stochastic confusion matrix ``P(o | y)``."""
    with np.load(prior_path(name, ".npz")) as data:
        confusion = np.asarray(data["confusion"], dtype=np.float64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError(
            f"Confusion matrix '{name}' must be square, got "
            f"{confusion.shape}."
        )
    if not np.allclose(confusion.sum(-1), 1.0, atol=1e-4):
        raise ValueError(f"Rows of confusion matrix '{name}' must sum to 1.")
    return confusion


class StructObservationLayer(torch.nn.Module):
    """Maps observed structural tokens to likelihoods over true tokens.

    ``M = (1 - s) I + s N`` where ``N`` is either the background
    distribution tiled over rows (``"background"``) or a confusion matrix
    (``"confusion"``). ``s`` is fixed or, if trainable, one scalar shared by
    all heads.

    Args:
        config: Structure configuration.
        strength: Initial mixing weight. Defaults to
            ``config.observation_noise_strength``; used to carry a trained
            value across model surgery.
    """

    def __init__(
        self,
        config: StructureConfig,
        strength: float | None = None,
    ) -> None:
        super().__init__()
        if config.observation_noise == "none":
            raise ValueError(
                "StructObservationLayer requires observation_noise != 'none'."
            )
        self.mode = config.observation_noise
        size = config.alphabet_size
        if self.mode == "background":
            pi = np.asarray(config.background_distribution, dtype=np.float64)
            pi = pi / pi.sum()
            noise = np.tile(pi, (size, 1))
        else:
            noise = load_confusion(config.observation_confusion_name)
        if noise.shape != (size, size):
            raise ValueError(
                f"Noise matrix has shape {noise.shape}, expected "
                f"{(size, size)} for the structural alphabet."
            )
        self.register_buffer(
            "noise", torch.as_tensor(noise, dtype=torch.float32)
        )
        self.register_buffer("eye", torch.eye(size, dtype=torch.float32))

        s = config.observation_noise_strength if strength is None \
            else float(strength)
        self.trainable = config.trainable_observation_noise
        if self.trainable:
            s = min(max(s, _MIN_STRENGTH), 1.0 - _MIN_STRENGTH)
            logit = float(np.log(s / (1.0 - s)))
            self.strength_logit = torch.nn.Parameter(
                torch.tensor(logit, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "fixed_strength", torch.tensor(s, dtype=torch.float32)
            )

    def strength(self) -> torch.Tensor:
        """The mixing weight ``s`` as a scalar tensor."""
        if self.trainable:
            return torch.sigmoid(self.strength_logit)
        return self.fixed_strength

    def matrix(self) -> torch.Tensor:
        """The observation matrix ``M[y, o] = P(o | y)``."""
        s = self.strength()
        return (1.0 - s) * self.eye + s * self.noise

    def num_parameters(self) -> int:
        """Number of free parameters (for model selection criteria)."""
        return 1 if self.trainable else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns ``v[..., y] = sum_o x[..., o] M[y, o]``."""
        return x.to(torch.float32) @ self.matrix().T


class StructLogitObservationLayer(torch.nn.Module):
    """Maps per-residue 3Di logits of a predictor to observation vectors.

    For logits z over the true letters (columns in structural alphabet
    order) the observation vector v, which anc-probs and the emitters use
    like a one-hot input, is

    - ``argmax``: the one-hot of ``argmax z``,
    - ``posterior``: ``softmax(z / T)``,
    - ``likelihood``: ``(softmax(z / T) / pi) ** k``, scaled to a maximum of
      1 per residue. Dividing the predictor's posterior by the letter prior
      pi gives a likelihood up to a per-residue constant, which does not
      change posteriors or alignments.

    Args:
        config: Structure configuration (``soft_input``,
            ``soft_input_temperature``, ``soft_input_sharpness``,
            ``background_distribution``).
    """

    def __init__(self, config: StructureConfig) -> None:
        super().__init__()
        self.mode = config.soft_input
        self.temperature = float(config.soft_input_temperature)
        self.sharpness = float(config.soft_input_sharpness)
        pi = np.asarray(config.background_distribution, dtype=np.float64)
        pi = pi / pi.sum()
        self.register_buffer(
            "log_prior", torch.as_tensor(np.log(pi), dtype=torch.float32)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(torch.float32)
        if self.mode == "argmax":
            return torch.nn.functional.one_hot(
                z.argmax(-1), z.shape[-1]
            ).to(torch.float32)
        log_post = torch.log_softmax(z / self.temperature, dim=-1)
        if self.mode == "posterior":
            return log_post.exp()
        score = self.sharpness * (log_post - self.log_prior)
        return (score - score.amax(-1, keepdim=True)).exp()
