"""Backend-neutral half of the all-zeros stand-in language model.

``zeros`` produces constant zero embeddings of a configurable width. It exists
so that the embedding pipeline can be exercised -- in tests and when only the
plain profile HMM is wanted -- without downloading a multi-gigabyte checkpoint.
The model wrappers live in :mod:`learnMSA.protein_language_models.tf.zeros` and
:mod:`learnMSA.protein_language_models.torch.zeros`.
"""

import sys
from typing import Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np

from learnMSA.protein_language_models.common import InputEncoder


class ZerosInputEncoder(InputEncoder):
    """Produces the padding mask a :class:`ZerosLanguageModel` needs.

    The token ids are zeros as well: nothing reads them, they only keep the
    encoder's output shape identical to the real encoders'.
    """

    @override
    def __call__(
        self, str_seq: Sequence[str], crop: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        del crop  # no special tokens, so nothing to crop
        lens = np.asarray([len(seq) for seq in str_seq], dtype=np.int32)
        max_len = int(np.max(lens, initial=0))
        ids = np.zeros((len(str_seq), max_len), dtype=np.int32)
        mask = np.zeros((len(str_seq), max_len), dtype=np.int32)
        for index, seq_len in enumerate(lens):
            mask[index, :seq_len] = 1
        return ids, mask
