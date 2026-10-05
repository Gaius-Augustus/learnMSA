from pathlib import Path

import numpy as np

from learnMSA.util.embedding_cache import EmbeddingCache

from .dataset import Dataset

# The keys of every embedding file. Any other key is metadata.
DEFAULT_KEYS = ("cache", "seq_lens", "seq_ids", "permutation", "dim")


class EmbeddingDataset(Dataset):
    """
    Manages a set of embeddings that are stored in a file or provided via an
    embedding cache.

    Attributes:
        filepath (Path): The path to the embedding file.
        num_seq (int): The number of sequences in the dataset.
        seq_lens (np.ndarray): An array containing the lengths of each sequence.
        max_len (int): The length of the longest sequence in the dataset.
        seq_ids (list[str]): The list of sequence IDs.
        parsing_ok (bool): Whether the dataset was parsed successfully.
        metadata (dict[str, np.ndarray]): Keys of the file other than
            ``DEFAULT_KEYS``. Empty for datasets built from a cache.
    """

    def __init__(
        self,
        filepath: Path | str | None = None,
        embedding_cache: EmbeddingCache | None = None,
        seq_ids: list[str] | None = None,
    ) -> None:
        """
        Args:
            filepath (Path): Path to an embedding file in any supported format.
                If given, embedding_cache is ignored and the dataset is
                initialized from the file.
            embedding_cache (EmbeddingCache): The embedding cache to use. Must
                be provided if filepath is not given. If given, the dataset is
                initialized from the cache.
            seq_ids (list[str]): Optional list of sequence IDs. Needs to be
                provided if embedding_cache is given.
        """
        self.filepath = Path(filepath) if filepath is not None else Path()
        self.fmt = ".npz"
        self.parsing_ok = False
        self.seq_ids: list[str] = []
        self.seq_lens = np.array([])
        self.metadata: dict[str, np.ndarray] = {}
        if filepath is None:
            assert embedding_cache is not None,\
                "Either filepath or embedding_cache must be provided."
            assert seq_ids is not None,\
                "seq_ids must be provided if embedding_cache is given."
            self.parsing_ok = True
            self._embedding_cache = embedding_cache
            self.seq_ids = seq_ids
            self.seq_lens = embedding_cache.seq_lens
            self._permutation = np.arange(self.seq_lens.size)
        else:
            self._parse_embedding_file(filepath)

        self.num_seq = len(self.seq_ids)
        if self.seq_lens.size > 0:
            self.max_len = int(np.amax(self.seq_lens))
        else:
            self.max_len = 0

    def close(self) -> None:
        pass

    def get_encoded_seq(
        self,
        i: int,
        crop_start: int | None = None,
        crop_end: int | None = None,
        dtype: type[np.integer | np.floating] = np.float32,
    ) -> np.ndarray:
        embedding = self._embedding_cache.get_embedding(self._permutation[i])
        if crop_end is not None:
            embedding = embedding[:crop_end]
        if crop_start is not None:
            embedding = embedding[crop_start:]
        return embedding.astype(dtype)

    def empty(
        self,
        shape: tuple[int, ...],
        dtype: type[np.integer | np.floating] = np.float32,
    ) -> np.ndarray:
        return np.zeros(shape + (self._embedding_cache.dim,), dtype=dtype)

    def write(
        self,
        filepath: Path | str,
        fmt="npz", # not used
        standardize_sequences: bool = False, # not used
        metadata: dict[str, str | np.ndarray] | None = None,
    ) -> Path:
        """
        Write the dataset to a binary ``.npz`` file.

        The file stores the embedding cache, sequence lengths, sequence IDs
        and the current permutation using NumPy's compressed npz format.

        Args:
            filepath (Path): Path to the output file. ``.npz`` is appended
                if missing.
            fmt (str): Unused, kept for interface compatibility.
            standardize_sequences (bool): Unused, kept for interface
                compatibility.
            metadata (dict): Extra keys to store. They are read back into
                ``metadata``. Must not use any of ``DEFAULT_KEYS``.

        Returns:
            The path written.
        """
        metadata = metadata or {}
        reserved = [k for k in metadata if k in DEFAULT_KEYS]
        if reserved:
            raise ValueError(f"Reserved metadata keys: {reserved}.")
        filepath = Path(filepath)
        if filepath.suffix != ".npz":
            filepath = filepath.with_suffix(filepath.suffix + ".npz")
        np.savez(
            filepath,
            cache=self._embedding_cache.cache,
            seq_lens=self._embedding_cache.seq_lens,
            seq_ids=np.array(self.seq_ids, dtype=str),
            permutation=self._permutation,
            dim=np.array([self._embedding_cache.dim]),
            **{k: np.asarray(v) for k, v in metadata.items()},
        )
        return filepath

    def reorder(self, permutation: list[int] | np.ndarray) -> None:
        """
        Reorder sequences in-place using a permutation of sequence indices.

        Args:
            permutation: A 1D permutation containing each index in
                [0, num_seq - 1] exactly once.
        """
        perm = np.asarray(permutation, dtype=np.int64)
        self.seq_ids = [self.seq_ids[i] for i in perm]
        self.seq_lens = self.seq_lens[perm]
        self._permutation = self._permutation[perm]

    def reorder_dims(self, order: list[int] | np.ndarray) -> None:
        """
        Select and reorder embedding dimensions in-place.

        Args:
            order: Indices of the old dimensions, in their new order.
        """
        cache = self._embedding_cache
        self._embedding_cache = EmbeddingCache(
            cache.seq_lens, len(order), cache=cache.cache[:, order]
        )

    def get_dtype(self) -> type[np.integer | np.floating]:
        """Return the dtype of the encoded sequences."""
        return np.float32

    def sample_embedding_variance(self, n_samples: int = 10000) -> np.ndarray:
        """ Approximates the variance of embedding dimensions. """
        if not self._embedding_cache.is_filled():
            raise ValueError(
                "Cannot sample embedding variance before embeddings are "
                + "computed."
            )
        return np.var(self._embedding_cache.cache[:n_samples], axis=0)


    def _parse_embedding_file(self, filepath: Path | str) -> None:
        """Parse an ``.emb`` file and initialize the dataset.

        The file is expected to be a NumPy npz archive containing the keys
        ``cache``, ``seq_lens``, ``seq_ids``, ``permutation`` and ``dim``.
        Any other keys are kept in ``metadata``.
        """
        filepath = Path(filepath)
        # np.savez may append .npz; accept both the original path and with .npz
        if not filepath.exists():
            npz_path = filepath.with_suffix(filepath.suffix + ".npz")
            if npz_path.exists():
                filepath = npz_path
        with np.load(filepath, allow_pickle=False) as data:
            seq_lens = data["seq_lens"]
            dim = int(data["dim"][0])
            cache_array = data["cache"]
            self.seq_ids = data["seq_ids"].tolist()
            self._permutation = data["permutation"]
            self.metadata = {
                k: data[k] for k in data.files if k not in DEFAULT_KEYS
            }
        self._embedding_cache = EmbeddingCache(seq_lens, dim, cache=cache_array)
        self.seq_lens = seq_lens[self._permutation]
        self.parsing_ok = True