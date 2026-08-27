from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Sequence, TypeVar

import numpy as np

#: Package-relative directory holding the shipped scoring model weights.
SCORING_MODEL_PATH = "scoring_models"

#: Embedding dimension each supported language model produces. ``zeros`` is a
#: stand-in whose dimension is chosen by the caller; the entry is its default.
dims: dict[str, int] = {
    "proteinBERT": 1562,
    "esm2": 2560,
    "protT5": 1024,
    "zeros": 16,
}

#: Language models that only the TensorFlow backend can run. ProteinBERT ships
#: as a Keras model in the ``proteinbert`` package and has no PyTorch port.
TF_ONLY_LANGUAGE_MODELS: tuple[str, ...] = ("proteinBERT",)

#: The tensor type of the selected backend.
T_Tensor = TypeVar("T_Tensor")


@dataclass(frozen=True)
class ScoringModelConfig:
    """Identifies one of the shipped bilinear scoring models.

    Attributes:
        lm_name: Name of the language model the scoring model was fitted for.
        dim: Reduced embedding dimension the scoring model projects to.
        activation: Output activation, ``"sigmoid"`` or ``"softmax"``.
        scaled: Whether scores are rescaled to roughly unit variance.
        suffix: Optional suffix identifying a non-default scoring model.
    """

    lm_name: str = "protT5"
    dim: int = 16
    activation: str = "sigmoid"
    scaled: bool = False
    suffix: str = ""


def get_scoring_model_path(config: ScoringModelConfig) -> str:
    """Package-relative path of a scoring model's legacy Keras weight file.

    Kept for the legacy pretraining tooling, which still writes ``.h5``. The
    production loader goes through
    :mod:`learnMSA.protein_language_models.scoring_weights`.
    """
    return (
        f"{SCORING_MODEL_PATH}/{config.lm_name}_{config.dim}_"
        f"{config.activation}{config.suffix}.h5"
    )


def make_cache_dir(path: str | Path | None, model_id: str) -> str:
    """Create and return the download cache directory for a language model.

    Args:
        path: Cache root. Defaults to ``~/.cache/learnmsa``.
        model_id: Subdirectory name identifying the model.

    Returns:
        The cache path for this model, as a string.
    """
    if path is None:
        path = Path.home() / ".cache" / "learnmsa"

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path / model_id)


class LanguageModel(Generic[T_Tensor]):
    """Base class for language models producing residue-level embeddings.

    The backend wrappers inherit framework-type-first, e.g.
    ``TorchLanguageModel(torch.nn.Module, LanguageModel[torch.Tensor])``. This
    class deliberately does not use ``ABCMeta``: it would clash with the keras
    and torch module metaclasses.
    """

    # These are annotations without values on purpose. A class-level ``= None``
    # would be found by ordinary attribute lookup and so shadow anything a
    # backend stores elsewhere -- ``torch.nn.Module`` keeps submodules in
    # ``_modules`` and only reaches them through ``__getattr__``, which is
    # never consulted while a class attribute of the same name exists.
    dim: int
    model: Any

    @abstractmethod
    def call(self, inputs: Sequence[Any]) -> T_Tensor:
        """Embed one batch of encoded sequences.

        Args:
            inputs: The tuple an :class:`InputEncoder` produced, already moved
                onto the backend's tensor type.

        Returns:
            Embeddings of shape ``(batch, max_len, dim)``, start- and
            end-tokens removed and trailing padding-only columns cropped.
        """

    @abstractmethod
    def eliminate_start_stop_tokens(
        self, embeddings: T_Tensor, crop: T_Tensor, mask: T_Tensor
    ) -> T_Tensor:
        """Strip the tokenizer's start- and end-tokens from a padded batch.

        Runs inside the compiled embedding call, so it must be written in the
        backend's own tensor ops -- hence one implementation per backend.

        Args:
            embeddings: Shape ``(batch, max_len, dim)``.
            crop: Shape ``(batch, 2)``, ``1`` where the sequence was cropped at
                the start / at the end and therefore carries no such token.
            mask: Shape ``(batch, max_len)``, ``1`` on non-padding positions.

        Returns:
            Embeddings with both special tokens removed, left-aligned, and
            trailing all-padding columns dropped.
        """

    def clear_internal_model(self) -> None:
        """Release the wrapped framework model."""
        if hasattr(self, "model"):
            del self.model


class InputEncoder:
    """Base class for encoders mapping protein strings to model inputs.

    Encoders are backend-neutral: they return numpy arrays, and the backend's
    embedding call converts them to its own tensor type.
    """

    @abstractmethod
    def __call__(
        self, str_seq: Sequence[str], crop: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Encode a batch of sequences.

        Args:
            str_seq: The sequences as plain strings.
            crop: Shape ``(batch, 2)`` of booleans, ``True`` where the sequence
                was cropped at the start / at the end.

        Returns:
            The tensors the matching :class:`LanguageModel` expects, as numpy
            arrays. The dtypes are set explicitly, because the backends
            convert them verbatim.
        """

    def modify_cropped(
        self,
        x: np.ndarray,
        crop: np.ndarray,
        lens: Sequence[int],
        pad_id: int,
    ) -> None:
        """Drop the special tokens of cropped sequences, in place.

        A sequence cropped at the start carries no start token, so everything
        shifts left by one; a sequence cropped at the end carries no end token.

        Args:
            x: Token ids or mask of shape ``(batch, max_len)``, modified in
                place.
            crop: Shape ``(batch, 2)`` of booleans.
            lens: Unpadded length of each sequence.
            pad_id: Value written into the freed positions.
        """
        for i, (cs, ce) in enumerate(crop):
            if cs:
                x[i] = np.roll(x[i], -1)
                x[i, -1] = pad_id
                if ce:
                    x[i, lens[i]] = pad_id
            elif ce:
                x[i, lens[i] + 1] = pad_id


def get_language_model(
    name: str,
    max_len: int = 512,
    trainable: bool = False,
    cache_dir: str | Path | None = None,
    embedding_dim: int | None = None,
) -> tuple[LanguageModel, InputEncoder]:
    """Construct a language model and its input encoder for the active backend.

    The wrapper class comes from the backend subpackage via
    :func:`learnMSA.backend.resolve`; the encoder is backend-neutral and is
    constructed directly.

    Args:
        name: One of ``"proteinBERT"``, ``"esm2"``, ``"esm2s"``, ``"protT5"``
            or ``"zeros"``.
        max_len: Maximum sequence length. Only ProteinBERT needs it.
        trainable: Whether the language model's weights are trainable.
        cache_dir: Where to cache the downloaded model.
        embedding_dim: Embedding width of the ``"zeros"`` stand-in.

    Returns:
        The language model and its input encoder.

    Raises:
        ValueError: If ``name`` is not a supported language model.
        NotImplementedError: If ``name`` is TensorFlow-only and another backend
            is active.
    """
    from learnMSA import backend

    _check_backend_supports(name)

    if name == "proteinBERT":
        make_protein_bert = backend.resolve(
            "protein_language_models.protein_bert", "make_protein_bert"
        )
        return make_protein_bert(
            max_len=max_len + 2, trainable=trainable, cache_dir=cache_dir
        )

    if name in ("esm2", "esm2s"):
        from learnMSA.protein_language_models import esm2

        small = name == "esm2s"
        language_model = backend.resolve(
            "protein_language_models.esm2", "ESM2LanguageModel"
        )(trainable=trainable, small=small, cache_dir=cache_dir)
        return language_model, esm2.ESM2InputEncoder(
            small=small, cache_dir=cache_dir
        )

    if name == "protT5":
        from learnMSA.protein_language_models import prot_t5

        language_model = backend.resolve(
            "protein_language_models.prot_t5", "ProtT5LanguageModel"
        )(trainable=trainable, cache_dir=cache_dir)
        return language_model, prot_t5.ProtT5InputEncoder(cache_dir=cache_dir)

    if name == "zeros":
        from learnMSA.protein_language_models import zeros

        dim = dims["zeros"] if embedding_dim is None else embedding_dim
        language_model = backend.resolve(
            "protein_language_models.zeros", "ZerosLanguageModel"
        )(embedding_dim=dim)
        return language_model, zeros.ZerosInputEncoder()

    raise ValueError(f"Language model {name} not supported.")


def _check_backend_supports(name: str) -> None:
    """Fail early and with a useful message on a TensorFlow-only model."""
    if name not in TF_ONLY_LANGUAGE_MODELS:
        return

    from learnMSA.backend import get_backend

    if get_backend() == "tensorflow":
        return

    raise NotImplementedError(
        f"The '{name}' language model is only available with the tensorflow "
        f"backend, but the '{get_backend()}' backend is active. Either pick "
        "another language model, or precompute the embeddings once with "
        f"'learnMSA --backend tensorflow --save-emb <file> ...' and pass them "
        "to this run with '--load-emb <file>'."
    )
