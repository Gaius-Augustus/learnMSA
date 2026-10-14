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
    like a one-hot input, is the posterior over the letters divided by the
    background pi, scaled to a maximum of 1 per residue. The posterior
    depends on the input format (T: ``input_temperature``):

    - ``"prostt5_logits"``: ``softmax(z / T)``, so ``v`` is proportional to
      ``softmax(z / T) / pi``,
    - ``"pdb_logits"``: z is ``-d^2``; the posterior is
      ``softmax(-d^2 / T + log(pi))``, so ``v`` is proportional to
      ``exp(z / T)``.

    Args:
        config: Structure configuration (``input_format``,
            ``input_temperature``, ``background_distribution``).
    """

    def __init__(self, config: StructureConfig) -> None:
        super().__init__()
        self.divide_prior = config.input_format == "prostt5_logits"
        self.temperature = float(config.resolved_input_temperature())
        pi = np.asarray(config.background_distribution, dtype=np.float64)
        pi = pi / pi.sum()
        self.register_buffer(
            "log_prior", torch.as_tensor(np.log(pi), dtype=torch.float32)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(torch.float32)
        if self.divide_prior:
            # z = logit of P(true letter | predicted letter)
            # pi = P(true letter)
            # compute log P(predicted letter | true letter) with Bayes
            log_post = torch.log_softmax(z / self.temperature, dim=-1)
            score = log_post - self.log_prior
        else:
            # z = -d^2 (distances to 20 VAE 3Di centroids)
            score = z / self.temperature
        return (score - score.amax(-1, keepdim=True)).exp()
