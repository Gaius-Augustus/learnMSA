"""When compiling a T5 encoder (ProtT5, ProstT5) beats running it eagerly.

Compiling costs a fixed overhead up front and then makes every batch faster,
mostly in attention, so the saving per residue grows linearly with the
sequence length and the total saving with the sum of squared lengths.
"""

import numpy as np

#: Seconds that compiling the encoder costs up front. Measured for ProtT5 on
#: an RTX 3090 with a cold inductor cache (~60 s; ~25 s warm).
COMPILE_OVERHEAD = 60.0

#: Seconds that compiling saves per residue and per residue of sequence
#: length. Measured for ProtT5 on an RTX 3090 (6% faster at length 100, 13%
#: at 300, 27% at 600, 30% at 1000; 4,000 proteins of length ~1000 took
#: 505 s compiled from a cold cache and 557 s eagerly).
COMPILE_SAVING = 3e-8


def compile_pays_off(seq_lens: np.ndarray) -> bool:
    """Whether compiling a T5 encoder on a GPU pays off for these sequences.

    Args:
        seq_lens: Lengths of all sequences that will be embedded.

    Returns:
        True if the estimated saving exceeds the compile overhead.
    """
    seq_lens = np.asarray(seq_lens, dtype=np.float64)
    return COMPILE_SAVING * float(np.sum(seq_lens**2)) > COMPILE_OVERHEAD
