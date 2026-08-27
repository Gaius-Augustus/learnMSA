import sys
from typing import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np

from learnMSA.protein_language_models.common import (InputEncoder,
                                                     make_cache_dir)

#: The 3B-parameter ESM-2 checkpoint, used by default.
MODEL_CHECKPOINT = "facebook/esm2_t36_3B_UR50D"

#: The 650M-parameter ESM-2 checkpoint, selected with ``small=True``.
MODEL_CHECKPOINT_SMALL = "facebook/esm2_t33_650M_UR50D"

#: Name of the download cache subdirectory shared by both checkpoints.
CACHE_ID = "esm2"

#: Embedding width of each checkpoint.
DIMS: dict[str, int] = {MODEL_CHECKPOINT: 2560, MODEL_CHECKPOINT_SMALL: 1280}


def checkpoint(small: bool) -> str:
    """The checkpoint name for the requested ESM-2 size."""
    return MODEL_CHECKPOINT_SMALL if small else MODEL_CHECKPOINT


class ESM2InputEncoder(InputEncoder):
    """Tokenizes proteins for ESM-2.

    ESM-2 brackets a full protein with a start- and an end-token. A sequence
    that was cropped is not a full protein any more, so the respective token is
    removed again.
    """

    def __init__(
        self, small: bool = False, cache_dir: str | None = None
    ) -> None:
        """
        Args:
            small: Use the 650M checkpoint instead of the 3B one.
            cache_dir: Where to cache the downloaded tokenizer.
        """
        from transformers import AutoTokenizer, logging

        logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint(small), cache_dir=make_cache_dir(cache_dir, CACHE_ID)
        )

    @override
    def __call__(
        self, str_seq: Sequence[str], crop: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tokens = self.tokenizer.batch_encode_plus(
            str_seq, add_special_tokens=True, padding=True, return_tensors="np"
        )
        ids = np.asarray(tokens["input_ids"], dtype=np.int32)
        mask = np.asarray(tokens["attention_mask"], dtype=np.int32)
        lens = [len(s) for s in str_seq]
        self.modify_cropped(ids, crop, lens, self.tokenizer.pad_token_id)
        self.modify_cropped(mask, crop, lens, 0)
        return ids, mask, np.asarray(crop, dtype=np.float32)
