"""Backend-neutral half of the ProtT5 language model wrapper.

Holds the checkpoint name and the input encoder. The model wrappers live in
:mod:`learnMSA.protein_language_models.tf.prot_t5` and
:mod:`learnMSA.protein_language_models.torch.prot_t5`.
"""

import re
import sys
from typing import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np

from learnMSA.protein_language_models.common import (InputEncoder,
                                                     make_cache_dir)

#: The half-precision ProtT5 encoder checkpoint.
MODEL_CHECKPOINT = "Rostlab/prot_t5_xl_half_uniref50-enc"

#: Name of the download cache subdirectory.
CACHE_ID = "protT5_model"

#: Embedding width of the ProtT5 encoder.
DIM = 1024


class ProtT5InputEncoder(InputEncoder):
    """Tokenizes proteins for ProtT5.

    ProtT5 uses a relative position embedding, so cropped sequences need no
    special treatment and ``crop`` is ignored.
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        """
        Args:
            cache_dir: Where to cache the downloaded tokenizer.
        """
        from transformers import T5Tokenizer, logging

        logging.set_verbosity_error()
        self.tokenizer = T5Tokenizer.from_pretrained(
            MODEL_CHECKPOINT,
            do_lower_case=False,
            cache_dir=make_cache_dir(cache_dir, CACHE_ID),
        )

    @override
    def __call__(
        self, str_seq: Sequence[str], crop: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        del crop  # relative position embedding, cropping is transparent
        # ProtT5 expects whitespace between residues; uncommon amino acids are
        # not in its vocabulary and are mapped to X.
        spaced = [
            re.sub(r"[UZOB]", "X", " ".join(sequence)) for sequence in str_seq
        ]
        tokens = self.tokenizer.batch_encode_plus(
            spaced, add_special_tokens=True, padding=True, return_tensors="np"
        )
        return (
            np.asarray(tokens["input_ids"], dtype=np.int32),
            np.asarray(tokens["attention_mask"], dtype=np.int32),
        )
