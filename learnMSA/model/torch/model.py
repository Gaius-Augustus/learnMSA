"""The PyTorch learnMSA model.

Everything that is the same whichever framework runs it -- the batching and
epoch heuristics, the derived likelihood/AIC estimates, the console reporting
-- lives in the backend-neutral :class:`learnMSA.model.model.LearnMSAModel`.
This module adds the torch module type and the methods that touch tensors.

The one genuinely new piece compared to the TensorFlow backend is the training
loop: there is no ``keras.Model.fit`` to inherit, so :meth:`TorchLearnMSAModel.fit`
runs the epochs itself and returns a small ``History`` stand-in, which is what
the inherited ``_check_training_complete`` reads.
"""

import math
import sys
import time
from typing import Any, Literal, Sequence, overload

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import torch

import learnMSA.backend as backend
from learnMSA.align.decode_util import DecodeArrays
from learnMSA.hmm.torch.layer import TorchPHMMLayer as PHMMLayer
from learnMSA.model.bucketing import make_default_bucket_scheme
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.model import LearnMSAModel
from learnMSA.model.torch.training import make_dataset
from learnMSA.tree.torch.anc_probs_layer import TorchAncProbsLayer
from learnMSA.util.sequence_dataset import Dataset, SequenceDataset


class RunningMean:
    """The torch stand-in for ``keras.metrics.Mean``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.reset_state()

    def reset_state(self) -> None:
        self._total = 0.0
        self._count = 0

    def update_state(self, value) -> None:
        if isinstance(value, torch.Tensor):
            value = float(value.detach())
        self._total += float(value)
        self._count += 1

    def result(self) -> float:
        return self._total / self._count if self._count else 0.0


class History:
    """The subset of ``keras.callbacks.History`` that learnMSA reads.

    ``LearnMSAModel._check_training_complete`` inspects ``history.history``
    for a ``"loss"`` entry, so ``fit`` has to return something shaped like
    this whichever backend produced it.
    """

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {}

    def append(self, logs: dict[str, float]) -> None:
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)


class TorchLearnMSAModel(torch.nn.Module, LearnMSAModel[torch.Tensor]):
    """
    The main model class for LearnMSA, combining a pHMM layer with
    ancestral probability encoding.
    Provides methods for training, evaluation, and prediction.
    """

    phmm_layer: PHMMLayer

    def __init__(self, context: LearnMSAContext, **kwargs) -> None:
        """
        Initialize the LearnMSA model.

        Args:
            context: The context containing model configuration.
            **kwargs: Accepted for signature parity with the TensorFlow model
                and ignored; torch modules take no base-class options.
        """
        torch.nn.Module.__init__(self)

        self.context = context
        train_cfg = context.config.training
        tree_cfg = context.config.tree

        # Declared on the instance rather than the class so that torch's
        # submodule lookup is reachable; see learnMSA.hmm.layer for why.
        self.anc_probs_layer = None

        # Create the ancestral probabilities layer
        if tree_cfg.use_anc_probs and not train_cfg.no_aa:
            self.anc_probs_layer = TorchAncProbsLayer(
                heads=train_cfg.num_model,
                rates=context.num_seq,
                input_tracks=2 if context.config.structure.use_structure else 1,
                equilibrium_init=context.p_init,
                rate_init=context.t_init,
                exchangeability_init=context.R_init,
                exchangeability_delta_init=context.R_delta_init,
                trainable_equilibrium=tree_cfg.trainable_equilibrium,
                trainable_exchangeabilities=tree_cfg.trainable_exchangeabilities,
                exchangeability_l2=tree_cfg.exchangeability_l2,
                trainable_rates=tree_cfg.trainable_rates,
                shared_equilibrium=tree_cfg.shared_equilibrium,
                shared_exchangeabilities=tree_cfg.shared_exchangeabilities,
                clusters=context.clusters,
                time_reversed=True,
                num_components=tree_cfg.num_anc_probs_components,
                mixture_init=context.mix_init,
                scale_init=context.scale_init,
                low_rank=tree_cfg.low_rank,
            )

        self.phmm_layer = PHMMLayer(
            lengths=context.model_lengths,
            config=context.config.hmm,
            prior_config=context.config.hmm_prior,
            plm_config=context.config.language_model,
            struct_config=context.config.structure,
            use_prior=context.config.training.use_prior,
            trainable_insertions=train_cfg.trainable_insertions,
            aa_value_sets=context.aa_values,
            struct_value_sets=context.struct_values,
            emb_value_sets=context.emb_values,
            joint_aa_struct_value_sets=context.joint_values,
            no_aa=train_cfg.no_aa,
        )

        # Metrics trackers
        self.loss_tracker = RunningMean(name="loss")
        self.loglik_tracker = RunningMean(name="loglik")
        self.prior_tracker = RunningMean(name="prior")

        # Per-model metrics for evaluation mode
        num_models = context.config.training.num_model
        self.loglik_per_model_trackers = [
            RunningMean(name=f"loglik_model_{i}") for i in range(num_models)
        ]
        self.prior_per_model_trackers = [
            RunningMean(name=f"prior_model_{i}") for i in range(num_models)
        ]
        self._eval_mode = False

        self.encode_hmm_inputs = tree_cfg.use_anc_probs

        self.optimizer: torch.optim.Optimizer | None = None
        self._device = torch.device(
            "cuda" if backend.num_gpus() > 0 else "cpu"
        )

    @property
    def device(self) -> torch.device:
        """The device the model's parameters live on.

        TensorFlow places tensors implicitly; torch does not, so both the
        training and the prediction loop move each batch here.
        """
        return self._device

    def forward(
        self,
        inputs: tuple[torch.Tensor | np.ndarray, ...],
        training: bool | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs: Tuple of (sequences, ..., indices) where:
                   - sequences: shape (batch, seq_length, num_models)
                   - ...: additional inputs depending on configuration
                    (e.g., for language model)
                   - indices: shape (batch, num_models)
            training: Accepted for signature parity with the TensorFlow model;
                torch tracks the mode on the module itself.

        Returns:
            The layer output, whose shape depends on the pHMM's call mode.
        """
        if len(inputs) < 2:
            raise ValueError(
                "inputs must contain at least sequences and indices"
            )
        sequences, *_ = inputs

        # Keep track of the runtime batch sizes for more verbose OOM error
        # handling
        self.context.last_runtime_batch_size = int(sequences.shape[0])

        # Pass through encoder layers
        forward_seq, *adds = self.encode_batch(inputs, training=training)

        padding = 1 - forward_seq[:, :, :, -1:]
        forward_seq = forward_seq[:, :, :, :-1]

        output = self.phmm_layer(forward_seq, adds=adds, padding=padding)

        if self.phmm_layer.is_loglik_mode():
            output = output.squeeze(-1)

        return output

    #: The neutral base calls this ``call``; torch modules use ``forward``.
    call = forward

    def encode_batch(
        self,
        inputs: tuple[torch.Tensor | np.ndarray, ...],
        training: bool | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        Encodes a batch of sequences with the ancestral probabilities layer.

        Args:
            inputs: Tuple of (sequences, ..., indices) where:
                   - sequences: shape (batch, seq_length, num_models)
                   - ...: additional inputs depending on configuration
                   - indices: shape (batch, num_models)
            training: Unused; kept for signature parity.

        Returns:
            A tensor with the ancestral probabilities of the input sequences
            as inputs to the pHMM layer.
        """
        if len(inputs) < 2:
            raise ValueError(
                "inputs must contain at least sequences and indices"
            )
        sequences, *adds, indices = inputs

        # Broadcast in the number of heads if necessary
        if self.phmm_layer.head_subset is not None:
            n = len(self.phmm_layer.head_subset)
        else:
            n = self.phmm_layer.heads
        if sequences.shape[2] == 1 and n > 1:
            sequences = sequences.repeat(1, 1, n, 1)
            indices = indices.repeat(1, n)

        # The amino acid track already arrives as per-residue distributions
        # over the emission alphabet (padding positions are all-zero vectors).
        # Append the terminal/padding channel (1 - sum) so that padding maps to
        # the terminal symbol and real residues map to 0 there.
        sequences = sequences.to(torch.float32)
        terminal = 1.0 - sequences.sum(dim=-1, keepdim=True)
        sequences_onehot = torch.cat([sequences, terminal], dim=-1)
        anc_prob_inputs = [sequences_onehot]

        if self.context.config.structure.use_structure:
            anc_prob_inputs.append(adds[0])

        if self.context.config.tree.use_anc_probs \
                and not self.context.config.training.no_aa \
                and self.encode_hmm_inputs:
            # AncProbsLayer accepts (batch, L, num_models, 20) and returns
            # (batch, L, num_models, num_matrices*20)
            encoded_seq = self.anc_probs_layer(
                *anc_prob_inputs, rate_indices=indices
            )
            if self.context.config.structure.use_structure:
                encoded_struct = encoded_seq[1]
                encoded_seq = encoded_seq[0]
                adds = [encoded_struct, *adds[1:]]
        else:
            encoded_seq = sequences_onehot
            # keep original adds; the structural track arrives one-hot encoded
            # from its SequenceDataset (remap=False)

        return encoded_seq, *adds

    def build(
        self, input_shapes: tuple[tuple[int | None, ...], ...] = ((None,),)
    ) -> None:
        """
        Build the model based on the input shape.

        Args:
            input_shapes: Shapes of the input data.
        """
        B = input_shapes[0][0]
        cfg = self.context.config
        s = cfg.hmm.alphabet_size
        n = self.phmm_layer.heads
        if self.anc_probs_layer is not None:
            self.anc_probs_layer.build()

        # Build the pHMM layer
        if cfg.training.no_aa:
            input_shape: tuple = ()
        else:
            input_shape = ((B, None, n, s),)
        if cfg.structure.use_structure:
            input_shape += ((B, None, n, cfg.structure.alphabet_size),)
        if cfg.language_model.use_language_model:
            emb_dim = cfg.language_model.scoring_model_dim
            input_shape += ((B, None, n, emb_dim),)
        input_shape += ((B, None, n, 1),)  # padding
        self.phmm_layer.build(input_shape=input_shape)

        # Only now do the parameters exist, so this is the earliest point at
        # which the whole model can be placed on a device.
        self.to(self._device)

    @override
    def compile(self, total_steps: int | None = None) -> None:
        """
        Set up the Adam optimizer.

        Args:
            total_steps: Accepted for signature parity with the TensorFlow
                model, which uses it to decide on JIT compilation.
        """
        self.optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.context.config.training.learning_rate,
        )

    @override
    def fit(
        self,
        data: SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
        indices: np.ndarray | None = None,
        iteration: int = 0,
        batch_size: int | None = None,
        epochs: int | None = None,
        steps_per_epoch: int | None = None,
        callbacks: list = [],
    ) -> History:
        """
        Fit the LearnMSA model on the specified sequences.

        Args:
            data: SequenceDataset or tuple of Dataset(s) with the first dataset
                being a SequenceDataset.
            indices: Array of sequence indices to train on.
            iteration: Current iteration number in the training loop.
            batch_size: Number of sequences per batch.
            epochs: Number of epochs to train for (overrides automatic
                setting).
            steps_per_epoch: Number of steps per epoch (overrides automatic
                setting).
            callbacks: Accepted for signature parity with the TensorFlow
                model; the torch loop has no callback protocol.

        Returns:
            A History object containing the per-epoch loss and metrics.
        """
        data = self._pack_datasets(data, "fit")

        self.phmm_layer.loglik_mode()
        self.context.batch_gen.configure(data, context=self.context)

        if batch_size is None:
            batch_size = self.get_batch_size(data[0])
        # Limit the training batch size to avoid convergence issues
        batch_size = min(batch_size, 512)

        if indices is None:
            indices = np.arange(data[0].num_seq)

        if epochs is None:
            epochs = self.get_num_epochs(iteration)

        if steps_per_epoch is None:
            steps_per_epoch = self.get_num_steps(indices.shape[0], batch_size)

        self.context.batch_gen.static_shape_mode = False
        self.compile(total_steps=steps_per_epoch)

        self._print_train_header(indices, batch_size, data[0])
        loader, _ = make_dataset(
            indices,
            self.context.batch_gen,
            batch_size,
            shuffle=True,
        )

        history = History()
        clipnorm = self.context.config.training.gradient_clipnorm
        verbose = self.context.config.input_output.verbose
        self.train()
        batches = iter(loader)
        for epoch in range(epochs):
            self.reset_metrics()
            epoch_start = time.perf_counter()
            for _ in range(steps_per_epoch):
                batch = next(batches)
                loss = self._train_step(batch, clipnorm)
                if not math.isfinite(loss):
                    # The TF backend has TerminateOnNaNWithCheckpoint for this;
                    # stopping here leaves the NaN in the history, which
                    # _check_training_complete then reports.
                    break
            logs = {
                "loss": self.loss_tracker.result(),
                "loglik": self.loglik_tracker.result(),
                "prior": self.prior_tracker.result(),
            }
            history.append(logs)
            if verbose:
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"Epoch {epoch + 1}/{epochs} - {elapsed:.1f}s - "
                    + " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())
                )
            if not math.isfinite(logs["loss"]):
                break
        self.eval()

        self._check_training_complete(history)
        return history

    def _train_step(self, batch, clipnorm: float) -> float:
        """One optimizer step on a single batch. Returns the loss."""
        assert self.optimizer is not None, "compile() must run before fit()."
        x = tuple(t.to(self._device) for t in batch)
        self.optimizer.zero_grad(set_to_none=True)
        y_pred = self(x)
        loss = self.compute_loss(x, None, y_pred)
        loss.backward()
        if clipnorm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clipnorm)
        self.optimizer.step()
        return float(loss.detach())

    def compute_loss(
        self,
        x: Any,
        y: Any,
        y_pred: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the total loss which combines likelihood and prior.

        In training mode, returns scalar loss with averaged metrics.
        In evaluation mode, returns scalar loss and per-model metrics.
        """
        weighted_loglik = self.weighted_loglik(x, y_pred)  # (num_models,)
        log_prior = self.log_prior()  # (num_models,)

        weighted_loglik_mean = weighted_loglik.mean()
        log_prior_mean = log_prior.mean()
        loss = -weighted_loglik_mean - log_prior_mean

        # Always update scalar metrics
        self.loss_tracker.update_state(loss)
        self.loglik_tracker.update_state(weighted_loglik_mean)
        self.prior_tracker.update_state(log_prior_mean)

        # In eval mode, also update per-model trackers
        if self._eval_mode:
            for i in range(len(self.loglik_per_model_trackers)):
                self.loglik_per_model_trackers[i].update_state(
                    weighted_loglik[i]
                )
                self.prior_per_model_trackers[i].update_state(log_prior[i])

        # Collect the regularization penalties that Keras would gather into
        # Model.losses; torch has no such hook.
        loss = loss + self.regularization_loss()

        return loss

    def regularization_loss(self) -> torch.Tensor:
        """The sum of the sub-layers' L2 penalties."""
        loss = torch.zeros((), device=self._device)
        if self.anc_probs_layer is not None:
            loss = loss + self.anc_probs_layer.regularization_loss()
        return loss

    def weighted_loglik(self, x: Any, y_pred: torch.Tensor) -> torch.Tensor:
        """ Computes the weighted loglik for the given batch.

        Args:
            x: Input data tuple containing sequences and indices.
            y_pred: Predicted loglikelihoods from the model.

        Returns:
            Tensor of shape (num_models,) with the weighted loglik values.
        """
        # Unstack the inputs
        _sequences, *_adds, indices = x

        # Apply sequence weights if provided
        if self.context.sequence_weights is not None:
            weights = torch.as_tensor(
                self.context.sequence_weights,
                dtype=torch.float32,
                device=y_pred.device,
            )[indices.to(torch.int64)]
            weighted_y_pred = y_pred * weights
            weighted_y_pred = weighted_y_pred / weights.sum(dim=0)
        else:
            weighted_y_pred = y_pred / y_pred.shape[0]

        # Reduce over the sequence dimension
        return weighted_y_pred.sum(dim=0)  # (H,)

    def log_prior(self) -> torch.Tensor:
        """ Computes the logarithmic prior value of each underlying model.

        Returns:
            Tensor of shape (num_models,) with the log prior values.
        """
        if self.context.sequence_weights is not None:
            num_cluster = self.context.sequence_weights.sum()
            # Use a geometric interpolation between the number of clusters
            # and the actual sequence count to be more stable; because the
            # former can be small
            S = math.sqrt(num_cluster * self.context.num_seq)
        else:
            S = self.context.num_seq
        log_prior = self.phmm_layer.prior_scores()
        if S > 0:
            log_prior = log_prior / S
        return log_prior

    def reset_metrics(self) -> None:
        """Reset all metric trackers."""
        self.loss_tracker.reset_state()
        self.loglik_tracker.reset_state()
        self.prior_tracker.reset_state()
        for tracker in self.loglik_per_model_trackers:
            tracker.reset_state()
        for tracker in self.prior_per_model_trackers:
            tracker.reset_state()

    @property
    def metrics(self) -> list[RunningMean]:
        return [self.loss_tracker, self.loglik_tracker, self.prior_tracker]

    @overload
    def predict(
        self,
        data: SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
        indices: np.ndarray | None = ...,
        models: list[int] | None = ...,
        bucket_boundaries: Sequence[int | float] | None = ...,
        bucket_batch_sizes: Sequence[int] | None = ...,
        reduce: bool = ...,
        ragged_output: Literal[False] = ...,
    ) -> np.ndarray: ...

    @overload
    def predict(
        self,
        data: SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
        indices: np.ndarray | None = ...,
        models: list[int] | None = ...,
        bucket_boundaries: Sequence[int | float] | None = ...,
        bucket_batch_sizes: Sequence[int] | None = ...,
        reduce: bool = ...,
        ragged_output: Literal[True] = ...,
    ) -> list[tuple[np.ndarray, np.ndarray]]: ...

    def predict(
        self,
        data: SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
        indices: np.ndarray | None = None,
        models: list[int] | None = None,
        bucket_boundaries: Sequence[int | float] | None = None,
        bucket_batch_sizes: Sequence[int] | None = None,
        reduce: bool = False,
        ragged_output: bool = False,
    ) -> np.ndarray | list[tuple[np.ndarray, np.ndarray]] | DecodeArrays:
        """
        Computes predictions for all sequences specified by indices in data.
        Buckets sequences by length for efficiency and determines appropriate
        batch sizes from the `TrainingConfiguration` of the model.
        Specify a call mode before running this method, e.g. `viterbi_mode()`.

        See :meth:`learnMSA.model.tf.model.TFLearnMSAModel.predict` for the
        full description of the arguments and the shape of the result.
        """
        if self._decode_msa:
            raise NotImplementedError(
                "The fused decoding modes (viterbi_decode_mode, "
                "mea_decode_mode) are only implemented for the TensorFlow "
                "backend. Use viterbi_mode() or mea_mode() and decode with "
                "AlignmentModel.decode instead."
            )

        data = self._pack_datasets(data, "predict")

        if indices is None:
            indices = np.arange(data[0].num_seq)
        start_time = time.perf_counter()

        # restrict to specified models
        prev_head_subset = self.phmm_layer.head_subset
        self.phmm_layer.head_subset = models
        if self.anc_probs_layer is not None:
            self.anc_probs_layer.head_subset = models

        if models is None:
            _models = list(range(len(self.phmm_layer.lengths)))
        else:
            _models = models

        loader, steps, old_crop_long_seqs = self._make_predict_loader(
            data, indices, _models, bucket_boundaries, bucket_batch_sizes
        )

        assert steps > 0, \
            "Prediction dataset must have a positive, finite number of steps."

        is_posterior = self.phmm_layer.is_posterior_mode()
        is_decoding = self.phmm_layer.is_decoding_mode()

        self.eval()
        if reduce and is_posterior:
            # Special reduced posterior mode: accumulate state posteriors
            # online to avoid storing full arrays in memory
            Q = max(self.phmm_layer.states[m] for m in _models)
            H = len(_models)
            accumulated_posteriors = np.zeros((H, Q), dtype=np.float32)

            with torch.no_grad():
                for batch in loader:
                    y, _ = self.predict_step(batch)
                    # Sum over batch and sequence positions
                    summed = y.sum(dim=(0, 1)).cpu().numpy()
                    accumulated_posteriors += summed[..., :-1]  # drop terminal

            accumulated_posteriors /= len(indices)
            self._finish_predict(
                old_crop_long_seqs, prev_head_subset, start_time,
                len(indices), steps,
            )
            return accumulated_posteriors

        def _replace_padding(pred: np.ndarray) -> None:
            if is_posterior:
                pred[pred == -1] = 0
            elif is_decoding:
                for i, j in enumerate(_models):
                    c = self.phmm_layer.lengths[j]
                    pred[:, :, i][pred[:, :, i] == -1] = 2 * c + 2

        outputs: np.ndarray | list
        with torch.no_grad():
            if ragged_output and (is_posterior or is_decoding):
                # Ragged output: accumulate batches per padded_len, then trim
                # and concatenate each group. A single bucket can produce
                # batches with different padded_len values, so pre-allocating a
                # fixed-size array per bucket is unreliable.
                bucket_preds: dict[int, tuple[list, list]] = {}

                for batch in loader:
                    batch_pred_t, batch_idx_t = self.predict_step(batch)
                    batch_pred = batch_pred_t.cpu().numpy()
                    batch_idx = batch_idx_t.cpu().numpy()
                    del batch_pred_t, batch_idx_t

                    padded_len = batch_pred.shape[1]
                    acc = bucket_preds.setdefault(padded_len, ([], []))
                    acc[0].append(batch_pred)
                    acc[1].append(batch_idx)

                outputs = []
                for padded_len, (preds, idxs) in bucket_preds.items():
                    bucket_idx = np.concatenate(idxs, axis=0)
                    bucket_max_len = int(
                        np.amax(data[0].seq_lens[indices[bucket_idx]]) + 1
                    )
                    L = min(padded_len, bucket_max_len)
                    bucket_pred = np.concatenate(
                        [p[:, :L] for p in preds], axis=0
                    )
                    _replace_padding(bucket_pred)
                    outputs.append((bucket_pred, bucket_idx))
            else:
                # Pre-allocate the full output array and write each batch
                # directly at the original sequence positions, eliminating list
                # accumulation and the 2x peak-memory spike from a final
                # concatenate call.
                N = len(indices)
                outputs_arr: np.ndarray | None = None

                if is_posterior or is_decoding:
                    max_len = int(np.amax(data[0].seq_lens[indices]) + 1)

                for batch in loader:
                    batch_pred_t, batch_idx_t = self.predict_step(batch)
                    batch_pred = batch_pred_t.cpu().numpy()
                    batch_idx = batch_idx_t.cpu().numpy()
                    del batch_pred_t, batch_idx_t

                    if outputs_arr is None:
                        # First batch: infer dtype and shape, allocate once.
                        if is_posterior or is_decoding:
                            pad_val: float | int = (
                                -1.0 if is_posterior else -1
                            )
                            tail = batch_pred.shape[2:]  # dims after (B, L)
                            outputs_arr = np.full(
                                (N, max_len) + tail, pad_val,
                                dtype=batch_pred.dtype,
                            )
                        else:
                            # Loglik: (B, H) -> allocate (N, H)
                            outputs_arr = np.empty(
                                (N,) + batch_pred.shape[1:],
                                dtype=batch_pred.dtype,
                            )

                    # Write directly to the original-order positions -- no
                    # sort/reorder step needed afterwards.
                    if is_posterior or is_decoding:
                        L = min(batch_pred.shape[1], max_len)
                        outputs_arr[batch_idx, :L] = batch_pred[:, :L]
                    else:
                        outputs_arr[batch_idx] = batch_pred

                    del batch_pred

                if outputs_arr is None:
                    raise RuntimeError("No predictions were produced.")

                _replace_padding(outputs_arr)
                outputs = outputs_arr

        self._finish_predict(
            old_crop_long_seqs, prev_head_subset, start_time,
            len(indices), steps,
        )
        return outputs

    def _make_predict_loader(
        self,
        data,
        indices: np.ndarray,
        models: list[int],
        bucket_boundaries: Sequence[int | float] | None,
        bucket_batch_sizes: Sequence[int] | None,
    ):
        """Shared batching setup for predict and evaluate."""
        self.context.batch_gen.configure(data, context=self.context)
        # Don't use static shapes for prediction - we'll use bucketing
        self.context.batch_gen.static_shape_mode = False
        old_crop_long_seqs = self.context.batch_gen.crop_long_seqs
        # do not crop sequences
        self.context.batch_gen.crop_long_seqs = math.inf

        # Override the number of models from the config if necessary
        if self.phmm_layer.head_subset is not None:
            self.context.batch_gen.num_models = len(
                self.phmm_layer.head_subset
            )

        if bucket_boundaries is None or bucket_batch_sizes is None:
            bucket_boundaries, bucket_batch_sizes = make_default_bucket_scheme(
                indices=indices,
                batch_generator=self.context.batch_gen,
                model_lengths=[self.phmm_layer.lengths[m] for m in models],
                batch_size_impl_factor=self.context._get_impl_factor(True),
            )

        loader, steps = make_dataset(
            indices,
            self.context.batch_gen,
            shuffle=False,
            bucket_by_seq_length=True,
            bucket_boundaries=bucket_boundaries,  # type: ignore
            bucket_batch_sizes=bucket_batch_sizes,
        )

        self._print_predict_header(
            indices, bucket_boundaries, bucket_batch_sizes, steps
        )
        self.compile(total_steps=steps)
        return loader, steps, old_crop_long_seqs

    def _finish_predict(
        self,
        old_crop_long_seqs,
        prev_head_subset,
        start_time: float,
        num_sequences: int,
        steps: int,
    ) -> None:
        self.context.batch_gen.crop_long_seqs = old_crop_long_seqs
        self.phmm_layer.head_subset = prev_head_subset
        if self.anc_probs_layer is not None:
            self.anc_probs_layer.head_subset = prev_head_subset
        self._print_predict_timing(
            elapsed_seconds=time.perf_counter() - start_time,
            num_sequences=num_sequences,
            steps=steps,
        )

    def predict_step(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs one bucketed batch through the model.

        Args:
            batch: ``(sequences, *adds, indices, j)`` where j is the position
                of each sequence within the requested index array.

        Returns:
            ``(predictions, j)`` so the caller can restore the original order.
        """
        x = tuple(t.to(self._device) for t in batch)
        *inputs, j = x
        return self(tuple(inputs)), j

    def evaluate(
        self,
        data: SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
        indices: np.ndarray | None = None,
        models: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Computes loss and loglik for all sequences specified by
        indices in data.

        Returns:
            Dictionary with keys 'loss' (scalar), 'loglik' (per-model),
            and 'prior' (per-model).
        """
        data = self._pack_datasets(data, "evaluate")

        if indices is None:
            indices = np.arange(data[0].num_seq)
        start_time = time.perf_counter()

        self.loglik_mode()

        # restrict to specified models
        prev_head_subset = self.phmm_layer.head_subset
        self.phmm_layer.head_subset = models
        if self.anc_probs_layer is not None:
            self.anc_probs_layer.head_subset = models

        if models is None:
            _models = list(range(len(self.phmm_layer.lengths)))
        else:
            _models = models

        loader, steps, old_crop_long_seqs = self._make_predict_loader(
            data, indices, _models, None, None
        )

        # Enable eval mode and reset per-model trackers
        self._eval_mode = True
        self.reset_metrics()
        self.eval()

        with torch.no_grad():
            for batch in loader:
                x = tuple(t.to(self._device) for t in batch)
                *inputs, _j = x
                y_pred = self(tuple(inputs))
                self.compute_loss(tuple(inputs), None, y_pred)

        self._eval_mode = False

        result = {
            "loss": np.asarray(self.loss_tracker.result()),
            "loglik": np.array([
                tracker.result()
                for tracker in self.loglik_per_model_trackers
            ]),
            "prior": np.array([
                tracker.result()
                for tracker in self.prior_per_model_trackers
            ]),
        }

        self._finish_predict(
            old_crop_long_seqs, prev_head_subset, start_time,
            len(indices), steps,
        )
        return result

    def compute_null_model_log_probs(
        self,
        data: Dataset,
        background_dist: np.ndarray | None = None,
        transition_prob: float | None = None,
    ) -> np.ndarray:
        """ Computes the logarithmic likelihood of each sequence under the
            null model

             S ---> T
            |_^    |_^
             p

            where the emission probabilities of S are amino acid background
            frequencies and T is the terminal state.

            Args:
                data: Dataset containing the sequences to evaluate.
                background_dist: Optional background distribution over the
                    alphabet. If None, uses the prior background distribution
                    from the emitter. Must be a flat array of size
                    alphabet_size.
                transition_prob: Optional self-loop transition probability.
                    If None, uses M/(M+1) where M is the mean sequence length.

            Returns:
                Array of log probabilities for each sequence.
        """
        n = data.num_seq

        # Configure batch generator
        self.context.batch_gen.configure(data, context=self.context)
        self.context.batch_gen.static_shape_mode = False
        old_crop_long_seqs = self.context.batch_gen.crop_long_seqs
        self.context.batch_gen.crop_long_seqs = math.inf

        null_indices = np.arange(n)
        bucket_boundaries, bucket_batch_sizes = make_default_bucket_scheme(
            indices=null_indices,
            batch_generator=self.context.batch_gen,
            model_lengths=self.phmm_layer.lengths,
            # effectively doubles the batch size compared to training
            batch_size_impl_factor=0.1,
            max_batch_size_override=10000,
        )

        loader, _ = make_dataset(
            null_indices,
            self.context.batch_gen,
            shuffle=False,
            bucket_by_seq_length=True,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
        )

        # Prepare the background frequencies over the emission alphabet.
        if background_dist is None:
            # Use prior background distribution
            assert self.phmm_layer.profile_emitter is not None
            assert self.phmm_layer.profile_emitter.prior, \
                "Emitter needs a Dirichlet prior for null model computation."
            with torch.no_grad():
                dirichlet_alpha = (
                    self.phmm_layer.profile_emitter.prior.matrix().cpu().numpy()
                )
            # Shared, pick any head and state
            dirichlet_alpha = dirichlet_alpha[0, 0]
            _background_dist = dirichlet_alpha / np.sum(dirichlet_alpha)

            # The prior lives on the 20 standard AAs; pad to the emission
            # alphabet size (small mass for U/O) when they are modeled.
            D = self.context.config.hmm.alphabet_size
            if _background_dist.shape[0] < D:
                pad = np.full(
                    D - _background_dist.shape[0],
                    float(_background_dist.min()),
                )
                _background_dist = np.concatenate([_background_dist, pad])
                _background_dist /= np.sum(_background_dist)
        else:
            _background_dist = np.asarray(background_dist, dtype=np.float32)

        bg = torch.as_tensor(
            _background_dist, dtype=torch.float32, device=self._device
        )
        log_probs = np.zeros((n,))

        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self._device)
                batch_idx = batch[-1].cpu().numpy()
                # Emissions arrive as per-residue distributions; the null
                # likelihood of a residue is the mixture P = sum_a v_a * bg_a.
                v = x[:, :, 0].to(torch.float32)  # (batch, seq_length, D)
                p = (v * bg).sum(dim=-1)  # (batch, seq_length)
                # Padding positions are all-zero vectors -> contribute nothing.
                mask = v.sum(dim=-1) > 0
                logp = torch.where(
                    mask, torch.log(p + 1e-10), torch.zeros_like(p)
                )
                log_probs[batch_idx] = logp.sum(dim=1).cpu().numpy()

        self.context.batch_gen.crop_long_seqs = old_crop_long_seqs

        # Add transition log probs based on the target sequence lengths.
        # We assume a geometric distribution with the expected length equal to
        # the length of each target sequence.
        L = np.asarray(data.seq_lens, dtype=np.float32)
        if transition_prob is None:
            M = float(L.mean())
            trans_scores = (L - 1.0) * (math.log(M) - math.log(M + 1.0))
        else:
            trans_scores = (L - 1.0) * math.log(transition_prob)
        log_probs += trans_scores

        return log_probs

    def compute_consensus_score(self) -> np.ndarray:
        """ Computes a consensus score that rates how plausible each model is
            with respect to all other models.

        Returns:
            Array of shape (num_models,) with consensus scores.
        """
        num_models = self.context.config.training.num_model
        model_lengths = self.phmm_layer.lengths
        alphabet_size = self.context.config.hmm.alphabet_size

        # compute the match sequence of all models padded with terminal symbols
        match_seqs = np.zeros(
            (num_models, max(model_lengths) + 1, alphabet_size + 1),
            dtype=np.float32,
        )
        match_seqs[:, :, -1] = 1  # initialize with terminal symbols
        assert self.phmm_layer.profile_emitter is not None, \
            "Consensus score computation requires a profile emitter."
        emitter = self.phmm_layer.profile_emitter
        with torch.no_grad():
            matrix = emitter.matrix().cpu().numpy()
        for i, L in enumerate(model_lengths):
            match_seqs[i, :L] = matrix[i, 1:L + 1]
        # we need to tile the match sequences over the batch dimension because
        # each model should see all other models
        stacked = torch.as_tensor(
            np.stack([match_seqs] * num_models, axis=1), device=self._device
        )

        # We skip the anc probs for simplicity as this would require fitting
        # evolutionary times.
        with torch.no_grad():
            consensus_logliks = self.phmm_layer.hmm.likelihood_log(
                stacked[:, :, :, :-1],  # Remove terminal symbol for emissions
                1 - stacked[:, :, :, -1:],  # Padding mask
            )
            eye = torch.eye(num_models, device=consensus_logliks.device)
            consensus_logliks = consensus_logliks * (1 - eye)
            # Axis 1 means we reduce over the batch dimension rather than the
            # model dimension, so output i is the mean loglik if we input all
            # other match sequences to model i. Using axis 0 is not the same:
            # consider the case that all models have the same match sequence
            # but one allows many deletions or insertions. That model is the
            # outlier and should have the lowest score. With axis=0 the
            # likelihood of the outlier under all other models is high and the
            # other models are penalised instead.
            consensus_score = consensus_logliks.mean(dim=1)
        return consensus_score.cpu().numpy()

    def get_train_callbacks(self, iteration: int = 0) -> list:
        """No callback protocol exists for the hand-written training loop.

        NaN termination and early stopping are handled inline in :meth:`fit`.
        """
        return []

    def save(self, filepath) -> None:
        """Persist the model as a torch checkpoint."""
        from learnMSA.model.torch.checkpoint import save_model

        save_model(self, filepath)
