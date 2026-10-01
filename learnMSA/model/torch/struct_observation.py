"""Observation model for per-residue 3Di predictions.

Structural tokens such as 3Di are often predicted from sequence (e.g. by
ProstT5) rather than computed from a structure. Instead of the predictor's
most likely letter, learnMSA can take its per-residue logits and map them to
an observation vector over the residue's true letter. The
ancestral-probabilities layer and the emitters act on that vector exactly as
on a one-hot input, because both are linear in it.
"""

import numpy as np
import torch

from learnMSA.config.structure import StructureConfig


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
