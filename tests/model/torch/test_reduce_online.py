"""``language_model.reduce_online``.

The mode hands the model the language model's full-dimensional embeddings --
cached like any others -- and lets a trainable bottleneck reduce them, instead
of projecting them once through the frozen bilinear scoring model.

No real language model is loaded here. The config names ``protT5`` because the
pHMM's embedding emitter needs a shipped MVN prior and only the real language
models have one; the embeddings themselves are synthetic.
"""

import math

import numpy as np
import pytest
import torch

from learnMSA.align.align import align
from learnMSA.config import (Configuration, LanguageModelConfig, TrainingConfig,
                             TreeConfig)
from learnMSA.model.batch_generator import BatchGenerator
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.torch.model import TorchLearnMSAModel
from learnMSA.model.torch.training import make_dataset
from learnMSA.util import EmbeddingCache, EmbeddingDataset
from tests.embedding_data import SEQ_LENS, make_aa_dataset

#: Width of the unreduced embeddings, standing in for a real pLM's.
FULL_DIM = 64

#: Width the pHMM sees, i.e. the bottleneck.
REDUCED_DIM = 16


def _config(**lm_kwargs) -> Configuration:
    return Configuration(
        training=TrainingConfig(
            length_init=[5], num_model=1, max_iterations=1, epochs=[1, 1, 1],
        ),
        tree=TreeConfig(use_anc_probs=False),
        language_model=LanguageModelConfig(
            use_language_model=True,
            scoring_model_dim=REDUCED_DIM,
            reduce_online=True,
            **lm_kwargs,
        ),
    )


def _full_embedding_dataset(
    dim: int = FULL_DIM, dtype: type[np.floating] = np.float32
) -> EmbeddingDataset:
    """A cache at the language model's full width, not the reduced one."""
    rows = [
        (i + 1) * np.ones((length, dim), dtype=dtype)
        for i, length in enumerate(SEQ_LENS)
    ]
    cache = EmbeddingCache(SEQ_LENS, dim, cache=np.concatenate(rows, axis=0))
    return EmbeddingDataset(
        embedding_cache=cache,
        seq_ids=[f"seq_{i}" for i in range(len(SEQ_LENS))],
    )


def _build_model(config: Configuration, aa_dataset, emb_dataset):
    context = LearnMSAContext(config, aa_dataset)
    context.embedding_dim = int(emb_dataset.empty(()).shape[-1])
    model = TorchLearnMSAModel(context)
    model.build(((2,),))
    return context, model


def test_full_width_embeddings_reach_the_model_and_are_reduced() -> None:
    """The batch carries D, the pHMM receives R."""
    aa_dataset, emb_dataset = make_aa_dataset(), _full_embedding_dataset()
    config = _config()
    _, model = _build_model(config, aa_dataset, emb_dataset)

    batch_gen = BatchGenerator()
    batch_gen.configure((aa_dataset, emb_dataset), model.context)
    loader, _ = make_dataset(
        np.arange(len(SEQ_LENS)), batch_gen, batch_size=2, shuffle=False
    )
    batch = next(iter(loader))

    _sequences, embeddings, _indices = batch
    assert embeddings.shape[-1] == FULL_DIM

    inputs = tuple(t.to(model.device) for t in batch)
    _encoded, *adds = model.encode_batch(inputs)
    assert adds[-1].shape[-1] == REDUCED_DIM


def test_reconstruction_loss_is_added_to_the_total() -> None:
    aa_dataset, emb_dataset = make_aa_dataset(), _full_embedding_dataset()
    config = _config(reduction_loss_weight=2.0)
    _, model = _build_model(config, aa_dataset, emb_dataset)
    model.loglik_mode()

    batch_gen = BatchGenerator()
    batch_gen.configure((aa_dataset, emb_dataset), model.context)
    loader, _ = make_dataset(
        np.arange(len(SEQ_LENS)), batch_gen, batch_size=2, shuffle=False
    )
    x = tuple(t.to(model.device) for t in next(iter(loader)))

    y_pred, reconstruction = model._forward_with_reconstruction(x)
    loss = model.compute_loss(
        x, None, y_pred, reconstruction_loss=reconstruction
    )

    assert torch.isfinite(loss)
    assert reconstruction > 0
    # The tracked loss is the total; subtracting the reconstruction term must
    # leave exactly what the model would report without a bottleneck.
    assert model.reconstruction_tracker.result() == pytest.approx(
        float(reconstruction.detach()), rel=1e-5
    )


def test_forward_return_loss() -> None:
    aa_dataset, emb_dataset = make_aa_dataset(), _full_embedding_dataset()
    _, model = _build_model(_config(reduction_loss_weight=2.0),
                            aa_dataset, emb_dataset)
    model.loglik_mode()

    batch_gen = BatchGenerator()
    batch_gen.configure((aa_dataset, emb_dataset), model.context)
    loader, _ = make_dataset(
        np.array([2, 3]), batch_gen, batch_size=2, shuffle=False
    )
    x = tuple(t.to(model.device) for t in next(iter(loader)))

    _y_pred, reconstruction = model(x, return_reconstruction_loss=True)
    torch.testing.assert_close(
        reconstruction, model.embedding_reconstruction_loss(x)
    )


def test_padding_is_excluded_from_the_reconstruction_loss() -> None:
    """Padded positions must not count; the batch is heavily padded here."""
    aa_dataset, emb_dataset = make_aa_dataset(), _full_embedding_dataset()
    _, model = _build_model(_config(), aa_dataset, emb_dataset)

    batch_gen = BatchGenerator()
    batch_gen.configure((aa_dataset, emb_dataset), model.context)
    # Sequence 3 has length 4, sequence 2 length 17 -> lots of padding.
    loader, _ = make_dataset(
        np.array([2, 3]), batch_gen, batch_size=2, shuffle=False
    )
    x = tuple(t.to(model.device) for t in next(iter(loader)))
    embeddings = x[-2]

    masked = model.embedding_reconstruction_loss(x)
    unmasked = model.embedding_encoder.reconstruction_loss(embeddings)
    assert not torch.isclose(masked, unmasked)


def test_language_model_weights_are_not_trainable() -> None:
    """Only the bottleneck and the regular learnMSA layers may be optimized.

    The language model is never a submodule of the model -- it lives in the
    dataset -- so this also guards against it being pulled in accidentally.
    """
    aa_dataset, emb_dataset = make_aa_dataset(), _full_embedding_dataset()
    _, model = _build_model(_config(), aa_dataset, emb_dataset)
    model.compile()

    optimized = {
        id(p) for group in model.optimizer.param_groups for p in group["params"]
    }
    assert all(
        id(p) in optimized for p in model.embedding_encoder.parameters()
    )
    assert not any(
        "language_model" in name for name, _ in model.named_parameters()
    )


def test_bottleneck_is_updated_by_training() -> None:
    aa_dataset, emb_dataset = make_aa_dataset(), _full_embedding_dataset()
    _, model = _build_model(_config(), aa_dataset, emb_dataset)

    before = model.embedding_encoder.encoder.weight.detach().clone()
    history = model.fit(
        (aa_dataset, emb_dataset),
        indices=np.arange(len(SEQ_LENS)),
        batch_size=2,
        epochs=1,
        steps_per_epoch=2,
    )

    assert np.all(np.isfinite(history.history["loss"]))
    assert "rec" in history.history
    assert not torch.allclose(
        before, model.embedding_encoder.encoder.weight.detach()
    )


def test_half_precision_batches_train_the_bottleneck() -> None:
    """The batch reaches the model in the cache's dtype and the model casts up.

    Full-width embeddings dominate every batch, so they travel to the device in
    the half precision the language model cached them in. Nothing between the
    cache and ``TorchEmbeddingEncoder.reduce`` may assume float32.
    """
    aa_dataset = make_aa_dataset()
    emb_dataset = _full_embedding_dataset(dtype=np.float16)
    assert emb_dataset.get_dtype() == np.float16
    _, model = _build_model(_config(), aa_dataset, emb_dataset)

    batch_gen = BatchGenerator()
    batch_gen.configure((aa_dataset, emb_dataset), model.context)
    loader, _ = make_dataset(
        np.arange(len(SEQ_LENS)), batch_gen, batch_size=2, shuffle=False
    )
    _sequences, embeddings, _indices = next(iter(loader))
    assert embeddings.dtype == torch.float16

    before = model.embedding_encoder.encoder.weight.detach().clone()
    history = model.fit(
        (aa_dataset, emb_dataset),
        indices=np.arange(len(SEQ_LENS)),
        batch_size=2, epochs=2, steps_per_epoch=2,
    )

    assert np.all(np.isfinite(history.history["loss"]))
    assert np.all(np.isfinite(history.history["rec"]))
    assert not torch.allclose(
        before, model.embedding_encoder.encoder.weight.detach()
    )


def test_bottleneck_survives_model_surgery() -> None:
    """align() rebuilds the model each iteration; the encoder must carry over."""
    aa_dataset, emb_dataset = make_aa_dataset(), _full_embedding_dataset()
    config = _config()
    config.training.max_iterations = 2
    config.input_output.output_file = ""

    align((aa_dataset, emb_dataset), config)  # runs at least one surgery round
    # align stashes the trained bottleneck on the context it built internally;
    # what matters here is that a rebuilt model picks a stashed state back up.
    context = LearnMSAContext(config, aa_dataset)
    context.embedding_dim = FULL_DIM
    reference = TorchLearnMSAModel(context)
    state = {
        k: v.detach().numpy() + 1.0
        for k, v in reference.embedding_encoder.state_dict().items()
    }
    context.emb_encoder_state = state
    resumed = TorchLearnMSAModel(context)
    for key, value in state.items():
        torch.testing.assert_close(
            resumed.embedding_encoder.state_dict()[key],
            torch.as_tensor(value),
        )


def test_already_reduced_embeddings_are_rejected() -> None:
    """A cache written without reduce_online holds the reduced width.

    Feeding one back in would leave nothing for the bottleneck to reduce, so
    say why rather than failing later on a shape mismatch.
    """
    aa_dataset = make_aa_dataset()
    reduced = _full_embedding_dataset(dim=REDUCED_DIM)

    with pytest.raises(ValueError, match="full embedding width"):
        align((aa_dataset, reduced), _config())


def test_high_dimensional_embeddings_need_reduce_online() -> None:
    """Without the bottleneck they would reach the pHMM unreduced.

    The emitter cannot emit a 1024-dimensional multivariate normal, so it has
    to say why instead of failing later on a broadcast inside mvn_log_prob.
    """
    aa_dataset = make_aa_dataset()
    wide = _full_embedding_dataset(dim=1024)
    config = _config()
    config.language_model.reduce_online = False

    with pytest.raises(ValueError, match="reduced before"):
        align((aa_dataset, wide), config)
