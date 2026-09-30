"""Per-residue 3Di logits from ProstT5 (PyTorch).

ProstT5 (Heinzinger et al., NAR Genom. Bioinform. 2024) is a T5 encoder; a
small CNN on its per-residue embeddings predicts 20 3Di classes. This module
reproduces the inference that ``foldseek createdb --prostt5-model`` runs, but
keeps the logits instead of only the argmax letter:

- the same weights: ``Rostlab/ProstT5_fp16`` at the revision the foldseek
  GGUF was converted from, and the CNN of ``predict_3Di_encoderOnly.py``,
- the same tokens: ``<AA2fold>``, one token per uppercase residue (unknown
  letters become ``X``), ``</s>``,
- the same split of sequences longer than 1024 residues,
- sequences are batched by length instead of run one at a time.
"""

import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

#: Hugging Face model and the pinned revision. Foldseek's
#: ``prostt5-f16.gguf`` records this revision as ``general.finetune``.
PROSTT5_REPO = "Rostlab/ProstT5_fp16"
PROSTT5_REVISION = "07a6547d51de603f1be84fd9f2db4680ee535a86"

#: 3Di letter of each CNN output class. ``ss_mapping`` in ProstT5's
#: ``scripts/predict_3Di_encoderOnly.py`` and ``number_to_char`` in foldseek's
#: ``src/strucclustutils/ProstT5.cpp`` both map class i to this string's i-th
#: letter. It equals ``StructureConfig.structural_alphabet``.
PROSTT5_CLASS_ORDER = "ACDEFGHIKLMNPQRSTVWY"

#: Token ids of the pinned ``spiece.model`` / ``added_tokens.json``.
PAD_ID = 0
EOS_ID = 1
AA2FOLD_ID = 149
X_ID = 23
TOKEN_IDS = {
    "A": 3, "L": 4, "G": 5, "V": 6, "S": 7, "R": 8, "E": 9, "D": 10,
    "T": 11, "I": 12, "P": 13, "K": 14, "F": 15, "Q": 16, "N": 17, "Y": 18,
    "M": 19, "H": 20, "W": 21, "C": 22, "X": 23, "B": 24, "O": 25, "U": 26,
    "Z": 27,
}

#: Foldseek's ``prostt5SplitLength`` and ``MIN_SPLIT_LENGTH``.
SPLIT_LENGTH = 1024
MIN_SPLIT_LENGTH = 2

CNN_WEIGHTS = Path(__file__).parent / "weights" / "prostt5_3di_cnn.npz"
EMBEDDING_DIM = 1024


def default_cache_dir() -> Path:
    """Where the ProstT5 encoder is cached (``~/.cache/learnmsa/prostt5``)."""
    return Path.home() / ".cache" / "learnmsa" / "prostt5"


def tokenize(seq: str) -> np.ndarray:
    """Token ids of the residues of ``seq`` (without prefix and ``</s>``)."""
    return np.fromiter(
        (TOKEN_IDS.get(c, X_ID) for c in seq.upper()),
        dtype=np.int64, count=len(seq),
    )


def split_chunks(
    length: int, split_length: int = SPLIT_LENGTH
) -> list[tuple[int, int]]:
    """(start, end) of the pieces foldseek predicts separately.

    Mirrors ``structcreatedb.cpp``: sequences longer than ``split_length``
    are cut into ``length // split_length + 1`` consecutive pieces; the piece
    length shrinks so that the last one has at least ``MIN_SPLIT_LENGTH``
    residues. ``split_length = 0`` disables splitting.
    """
    if split_length <= 0 or length <= split_length:
        return [(0, length)]
    n_splits = length // split_length + 1
    remainder = length % split_length
    if remainder < MIN_SPLIT_LENGTH:
        split_length -= (MIN_SPLIT_LENGTH - remainder) // (n_splits - 1) + 1
    return [
        (i * split_length, min((i + 1) * split_length, length))
        for i in range(n_splits)
        if i * split_length < length
    ]


class ProstT5CNN(torch.nn.Module):
    """The 3Di head of ``predict_3Di_encoderOnly.py``."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(EMBEDDING_DIM, 32, kernel_size=(7, 1),
                            padding=(3, 0)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.0),
            torch.nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0)),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """(B, W, 1024) embeddings -> (B, W, 20) logits.

        Foldseek (and ``predict_3Di_encoderOnly.py`` at batch size 1) runs
        the head on the L residue embeddings followed by one zero row, so the
        hidden layer is nonzero at position L and zero beyond. With
        ``lengths``, positions past each sequence are masked the same way,
        which makes a batched sequence identical to one run alone. The caller
        must zero embeddings at positions >= L.
        """
        x = x.permute(0, 2, 1).unsqueeze(-1)
        hidden = self.classifier[1](self.classifier[0](x))
        if lengths is not None:
            positions = torch.arange(hidden.shape[2], device=hidden.device)
            keep = positions[None, :] <= lengths[:, None]
            hidden = hidden * keep[:, None, :, None]
        out = self.classifier[3](self.classifier[2](hidden))
        return out.squeeze(-1).permute(0, 2, 1)

    @classmethod
    def from_npz(cls, path: str | Path = CNN_WEIGHTS) -> "ProstT5CNN":
        model = cls()
        with np.load(path, allow_pickle=False) as data:
            state = {k: torch.as_tensor(data[k]) for k in data.files
                     if k.startswith("classifier.")}
        model.load_state_dict(state)
        return model.eval()


def load_encoder(
    device: torch.device, cache_dir: str | Path | None = None
) -> torch.nn.Module:
    """The ProstT5 encoder, fp16 on GPU and fp32 on CPU."""
    import transformers
    from packaging.version import Version
    from transformers import T5EncoderModel

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    # transformers 4.56 renamed torch_dtype to dtype.
    dtype_key = "dtype" if Version(transformers.__version__) >= \
        Version("4.56") else "torch_dtype"
    encoder = T5EncoderModel.from_pretrained(
        PROSTT5_REPO,
        revision=PROSTT5_REVISION,
        cache_dir=str(cache_dir or default_cache_dir()),
        # The pinned revision only has pytorch_model.bin. Without this,
        # transformers also downloads and prefers an unmerged bot
        # conversion to safetensors (refs/pr/1), which is not pinned.
        use_safetensors=False,
        **{dtype_key: dtype},
    )
    return encoder.to(device).eval()


def _bytes_per_sequence(length: int, config) -> float:
    """Rough peak activation memory of one sequence in the fp16 encoder.

    Eager T5 attention holds fp16 scores, a float32 softmax, its fp16 cast
    and the batch-sized position bias at once (about 12 bytes per head and
    token pair); the feed-forward block about three fp16 (tokens, d_ff)
    tensors.
    """
    tokens = length + 2
    attention = 12 * config.num_heads * tokens * tokens
    feed_forward = 6 * tokens * config.d_ff
    return attention + feed_forward + 16 * tokens * config.d_model


class ProstT5Predictor:
    """Predicts per-residue 3Di logits.

    Args:
        device: Torch device; defaults to CUDA if available.
        cache_dir: Hugging Face cache for the encoder.
        encoder: A prebuilt encoder (for tests); loaded if None.
        cnn: A prebuilt 3Di head (for tests); the shipped one if None.
        batch_memory: Activation memory (bytes, estimated) a batch may use;
            at most half of the free GPU memory. Larger batches are not
            faster: on an RTX 3090, 2 GiB and 11 GiB give the same speed.
        max_batch: Upper bound on the number of pieces per batch.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        encoder: torch.nn.Module | None = None,
        cnn: torch.nn.Module | None = None,
        batch_memory: float = 2 * 2**30,
        max_batch: int = 512,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.encoder = encoder if encoder is not None \
            else load_encoder(self.device, cache_dir)
        self.encoder = self.encoder.to(self.device).eval()
        self.cnn = (cnn if cnn is not None
                    else ProstT5CNN.from_npz()).to(self.device).eval()
        self.batch_memory = batch_memory
        self.max_batch = max_batch

    def _memory_budget(self) -> float:
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            return min(self.batch_memory, 0.5 * free)
        return self.batch_memory

    def _batch_size(self, length: int, budget: float) -> int:
        per_sequence = _bytes_per_sequence(length, self.encoder.config)
        return int(max(1, min(self.max_batch, budget // per_sequence)))

    @torch.inference_mode()
    def _forward(self, pieces: Sequence[np.ndarray]) -> torch.Tensor:
        """Logits (B, Lmax, 20), float32, for a batch of token pieces."""
        lengths = [len(p) for p in pieces]
        width = max(lengths) + 2
        ids = torch.full((len(pieces), width), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(pieces), width), dtype=torch.long)
        for b, piece in enumerate(pieces):
            ids[b, 0] = AA2FOLD_ID
            ids[b, 1:1 + len(piece)] = torch.as_tensor(piece)
            ids[b, 1 + len(piece)] = EOS_ID
            mask[b, :len(piece) + 2] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        hidden = self.encoder(
            input_ids=ids, attention_mask=mask
        ).last_hidden_state
        # Drop <AA2fold>; keep one row past the longest piece (the zero row
        # foldseek appends) and zero every position past each piece.
        hidden = hidden[:, 1:].float()
        lengths_t = torch.as_tensor(lengths, device=self.device)
        positions = torch.arange(hidden.shape[1], device=self.device)
        hidden = hidden * (positions[None, :] < lengths_t[:, None])[..., None]
        return self.cnn(hidden, lengths_t)[:, :width - 2]

    def predict(
        self,
        seqs: Sequence[str],
        split_length: int = SPLIT_LENGTH,
        out: np.ndarray | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Logits of all residues, concatenated in input order.

        Args:
            seqs: Amino acid sequences (gap-free).
            split_length: Foldseek's split length; 0 disables splitting.
            out: Optional preallocated (sum of lengths, 20) array to fill
                (e.g. float16 to save memory).
            progress: Called with (pieces done, pieces total).

        Returns:
            Array of shape (sum of lengths, 20).
        """
        lengths = np.array([len(s) for s in seqs], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        if out is None:
            out = np.zeros((int(offsets[-1]), 20), dtype=np.float32)
        pieces, targets = [], []
        for i, seq in enumerate(seqs):
            tokens = tokenize(seq)
            for start, end in split_chunks(len(seq), split_length):
                pieces.append(tokens[start:end])
                targets.append(offsets[i] + start)
        order = sorted(range(len(pieces)), key=lambda k: -len(pieces[k]))
        budget = self._memory_budget()
        done, pos = 0, 0
        while pos < len(order):
            length = len(pieces[order[pos]])
            if length == 0:
                break
            size = self._batch_size(length, budget)
            while True:
                batch = order[pos:pos + size]
                try:
                    logits = self._forward([pieces[k] for k in batch])
                    break
                except torch.cuda.OutOfMemoryError:
                    if size == 1:
                        raise
                    torch.cuda.empty_cache()
                    size = max(1, size // 2)
                    budget = min(budget, size * _bytes_per_sequence(
                        length, self.encoder.config))
            logits = logits.cpu().numpy()
            for b, k in enumerate(batch):
                n = len(pieces[k])
                out[targets[k]:targets[k] + n] = logits[b, :n]
            pos += len(batch)
            done += len(batch)
            if progress is not None:
                progress(done, len(pieces))
        return out


def predict_logits(
    seqs: Sequence[str],
    device: str | None = None,
    cache_dir: str | Path | None = None,
    split_length: int = SPLIT_LENGTH,
    dtype: type[np.floating] = np.float16,
    verbose: bool = True,
    batch_memory: float = 2 * 2**30,
) -> np.ndarray:
    """Convenience wrapper: loads ProstT5 and predicts all ``seqs``."""
    t0 = time.time()
    predictor = ProstT5Predictor(device=device, cache_dir=cache_dir,
                                 batch_memory=batch_memory)
    if verbose:
        print(f"Loaded ProstT5 on {predictor.device} "
              f"in {time.time() - t0:.1f} s.", flush=True)
    total = sum(len(s) for s in seqs)
    out = np.zeros((total, 20), dtype=dtype)
    last = [0]

    def report(done: int, n: int) -> None:
        step = math.floor(10 * done / n)
        if verbose and step > last[0]:
            last[0] = step
            print(f"{10 * step}% of {n} pieces", flush=True)

    t1 = time.time()
    predictor.predict(seqs, split_length=split_length, out=out,
                      progress=report)
    if verbose:
        elapsed = time.time() - t1
        print(f"Predicted {len(seqs)} sequences ({total} residues) in "
              f"{elapsed:.1f} s ({total / max(elapsed, 1e-9):.0f} "
              "residues/s).", flush=True)
    return out
