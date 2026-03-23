import math
from pathlib import Path
import time
from typing import Any, Literal, Sequence, override

import numpy as np
import tensorflow as tf

from learnMSA.hmm.tf.layer import PHMMLayer
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.tf.phmm_mixin import PHMMMixin
from learnMSA.model.tf.training import (TerminateOnNaNWithCheckpoint,
                                        make_dataset,
                                        make_default_bucket_scheme)
from learnMSA.tree.tf.anc_probs_layer import AncProbsLayer
from learnMSA.util.sequence_dataset import Dataset, SequenceDataset
from learnMSA.util.clustering import write_sequence_weights


class LearnMSAModel(tf.keras.Model, PHMMMixin):
    """
    The main model class for LearnMSA, combining a pHMM layer with
    ancestoral probability encoding.
    Provides methods for training, evaluation, and prediction.
    """

    phmm_layer: PHMMLayer

    # enable proper type checking for __new__
    # otherwise the tf type stubs package causes trouble
    def __new__(cls, *args, **kwargs) -> "LearnMSAModel":
        instance = super().__new__(cls)
        return instance # type: ignore[return-value]

    def __init__(self, context: LearnMSAContext, **kwargs) -> None:
        """
        Initialize the LearnMSA model.

        Args:
            context (LearnMSAContext): The context containing model
                configuration.
                Can be None only during deserialization.
            **kwargs: Additional keyword arguments for the base Model class
                (e.g., trainable, dtype, name). These are used during
                deserialization.
        """
        # Filter out base Model kwargs before calling super().__init__()
        super().__init__(**kwargs)

        self.context = context
        train_cfg = context.config.training

        # Create the ancestor probabilities layer
        if train_cfg.use_anc_probs and not train_cfg.no_aa:
            self.anc_probs_layer = AncProbsLayer(
                train_cfg.num_model,
                context.num_seq,
                train_cfg.num_rate_matrices,
                equilibrium_init=context.encoder_initializer[2],
                rate_init=context.encoder_initializer[0],
                exchangeability_init=context.encoder_initializer[1],
                trainable_rate_matrices=train_cfg.trainable_rate_matrices,
                trainable_distances=train_cfg.trainable_distances,
                matrix_rate_l2=train_cfg.matrix_rate_l2,
                shared_matrix=train_cfg.shared_rate_matrix,
                equilibrium_sample=train_cfg.equilibrium_sample,
                transposed=train_cfg.transposed,
                clusters=context.clusters
            )

        self.phmm_layer = PHMMLayer(
            lengths = context.model_lengths,
            config = context.config.hmm,
            prior_config = context.config.hmm_prior,
            plm_config = context.config.language_model,
            structural_config = context.config.structure,
            use_prior = context.config.training.use_prior,
            trainable_insertions = train_cfg.trainable_insertions,
            value_sets = context.init_msa_values,
        )

        # Metrics trackers
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.loglik_tracker = tf.keras.metrics.Mean(name='loglik')
        self.prior_tracker = tf.keras.metrics.Mean(name='prior')

        # Per-model metrics for evaluation mode
        num_models = context.config.training.num_model
        self.loglik_per_model_trackers = [
            tf.keras.metrics.Mean(name=f'loglik_model_{i}')
            for i in range(num_models)
        ]
        self.prior_per_model_trackers = [
            tf.keras.metrics.Mean(name=f'prior_model_{i}')
            for i in range(num_models)
        ]
        self._eval_mode = False

        self.encode_hmm_inputs = train_cfg.use_anc_probs

    def call(
        self,
        inputs: tuple[tf.Tensor | np.ndarray, ...],
        training: bool|None=None,
    ) -> tuple[tf.Tensor, ...]:
        """
        Forward pass of the model.

        Args:
            inputs: Tuple of (sequences, ..., indices) where:
                   - sequences: shape (batch, seq_length, num_models)
                   - ...: additional inputs depending on configuration
                    (e.g., for language model)
                   - indices: shape (batch, num_models)
            training: Boolean indicating training mode.

        Returns:
            Tuple of outputs depending on use_prior:
            - Without prior: (loglik, aggregated_loglik)
            - With prior: (loglik, aggregated_loglik, prior, aux_loss)
        """
        if len(inputs) < 2:
            raise ValueError(
                "inputs must contain at least sequences and indices"
            )
        sequences, *adds, _indices = inputs

        # Keep track of the runtime batch sizes for more verbose OOM error
        # handling
        if isinstance(sequences, tf.Tensor):
            B = sequences.shape[0]
            if B is None:
                B = tf.get_static_value(tf.shape(sequences)[0])
            if B is not None:
                self.context.last_runtime_batch_size = int(B)
        elif isinstance(sequences, np.ndarray):
            self.context.last_runtime_batch_size = int(sequences.shape[0])

        # Pass through encoder layers
        forward_seq = self.encode_batch(inputs, training=training)

        padding = 1 - forward_seq[:, :, :, -1:]
        forward_seq = forward_seq[:, :, :, :-1]

        output = self.phmm_layer(forward_seq, adds=adds, padding=padding)

        if self.phmm_layer.is_loglik_mode():
            output = tf.squeeze(output, axis=-1)

        return output

    def encode_batch(
        self,
        inputs: tuple[tf.Tensor | np.ndarray, ...],
        training: bool|None=None,
    ) -> tf.Tensor:
        """
        Encodes a batch of sequences with the ancestral probabilities layer.

        Args:
            inputs: Tuple of (sequences, ..., indices) where:
                   - sequences: shape (batch, seq_length, num_models)
                   - ...: additional inputs depending on configuration
                    (e.g., for language model) (not used here)
                   - indices: shape (batch, num_models)
            training: Boolean indicating training mode.

        Returns:
            A tensor with the ancestral probabilities of the input sequences
            as inputs to the pHMM layer.
        """
        if len(inputs) < 2:
            raise ValueError(
                "inputs must contain at least sequences and indices"
            )
        sequences, *_adds, indices = inputs

        # Broadcast in the number of heads if necessary
        if self.phmm_layer.head_subset is not None:
            n = len(self.phmm_layer.head_subset)
        else:
            n = self.phmm_layer.heads
        if sequences.shape[2] == 1 and n > 1:
            sequences = tf.tile(sequences, [1, 1, n])
            indices = tf.tile(indices, [1, n])

        # Convert to one-hot for AncProbsLayer (now requires 4D input)
        sequences_onehot = tf.one_hot(
            tf.cast(sequences, tf.int32),
            depth=self.context.config.hmm.alphabet_size+1, # including terminal
            dtype=self.phmm_layer.dtype
        )

        if self.context.config.training.use_anc_probs\
                and self.encode_hmm_inputs:
            # AncProbsLayer accepts (batch, L, num_models, 20) and returns
            # (batch, L, num_models, num_matrices*20)
            encoded_seq = self.anc_probs_layer(
                sequences_onehot, rate_indices=indices, training=training # type: ignore
            )
        else:
            encoded_seq = sequences_onehot

        return encoded_seq

    @override
    def build(
        self, input_shapes: tuple[tuple[int | None, ...], ...] = ((None,),)
    ) -> None:
        """
        Build the model based on the input shape.

        Args:
            input_shape: Shape of the input data.
        """
        B = input_shapes[0][0]
        cfg = self.context.config
        s = cfg.hmm.alphabet_size
        n = self.phmm_layer.heads
        # AncProbsLayer now expects (batch, L, num_models) shape
        seq_shape_batch_first = (B, None, n)
        ind_shape_batch_first = (B, n)
        if cfg.training.use_anc_probs:
            self.anc_probs_layer.build([seq_shape_batch_first, ind_shape_batch_first])

        # Build the pHMM layer
        input_shape = ((B, None, n, s),)
        if cfg.language_model.use_language_model:
            emb_dim = cfg.language_model.scoring_model_dim
            input_shape += ((B, None, n, emb_dim),)
        if cfg.structure.use_structure:
             input_shape += ((B, None, n, cfg.structure.alphabet_size),)
        input_shape += ((B, None, n, 1),) # padding
        self.phmm_layer.build(input_shape = input_shape)

    @override
    def compile(self, total_steps: int | None = None) -> None:
        """
        Compile the model with Adam optimizer.

        This method compiles the LearnMSA model using the Adam optimizer
        with the learning rate specified in the configuration.

        Args:
            total_steps: The total number of steps the model will be called
                (optional). If provided, it is used to decide if the model
                should be jit-compiled.

        """
        optimizer = tf.keras.optimizers.Adam(
            self.context.config.training.learning_rate
        )
        super().compile(
            optimizer=optimizer,
            jit_compile=self.use_jit_compile(total_steps),
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
        callbacks: list[tf.keras.callbacks.Callback]=[],
    ) -> tf.keras.callbacks.History:
        """
        Fit the LearnMSA model on the specified sequences.

        Args:
            data: SequenceDataset or tuple of Dataset(s) with the first dataset
                being a SequenceDataset.
                When multiple datasets are provided, they are
                required to have the same order (i.e. index i in each
                dataset corresponds to the same sequence) and the HMM must
                use a fitting emitter for each dataset. The first dataset
                will be used to query metadata such as the number of sequences
                and sequence lengths, although these metrics should be
                consistent across datasets.
            indices: Array of sequence indices to train on
            iteration: Current iteration number in the training loop
            batch_size: Number of sequences per batch
            epochs: Number of epochs to train for (overrides automatic setting)
            steps_per_epoch: Number of steps per epoch
                (overrides automatic setting)
            callbacks: List of Keras callbacks to use during training

        Returns:
            Training history object containing loss and metrics
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

        # use static shapes when JIT is enabled
        s = steps_per_epoch
        self.context.batch_gen.static_shape_mode = self.use_jit_compile(s)
        self.compile(total_steps=s)

        self._print_train_header(indices, batch_size, data[0])
        dataset, _ = make_dataset(
            indices,
            self.context.batch_gen,
            batch_size,
            shuffle=True,
        )
        history = super().fit(
            dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks + self.get_train_callbacks(),
            verbose=self.get_verbosity(),
        )
        self._check_training_complete(history)
        return history

    def compute_loss(
        self,
        x: Any,
        y: Any,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """
        Compute the total loss which combines likelihood and prior.

        In training mode, returns scalar loss with averaged metrics.
        In evaluation mode, returns scalar loss and per-model metrics.
        """
        weighted_loglik = self.weighted_loglik(x, y_pred)  # (num_models,)
        log_prior = self.log_prior()  # (num_models,)

        weighted_loglik_mean = tf.reduce_mean(weighted_loglik)
        log_prior_mean = tf.reduce_mean(log_prior)
        loss = -weighted_loglik_mean - log_prior_mean

        # Always update scalar metrics
        self.loss_tracker.update_state(loss)
        self.loglik_tracker.update_state(weighted_loglik_mean)
        self.prior_tracker.update_state(log_prior_mean)

        # In eval mode, also update per-model trackers
        if self._eval_mode:
            for i in range(len(self.loglik_per_model_trackers)):
                self.loglik_per_model_trackers[i].update_state(weighted_loglik[i])
                self.prior_per_model_trackers[i].update_state(log_prior[i])

        # Collect custom losses from layers
        loss += sum(self.losses)

        return loss

    def weighted_loglik(self, x: Any, y_pred: tf.Tensor) -> tf.Tensor:
        """ Computes the weighted loglik for the given batch.

        Args:
            x: Input data tuple containing sequences and indices.
            y_pred: Predicted loglikelihoods from the model.

        Returns:
            weighted_loglik: Tensor of shape (num_models,) with the weighted
                loglik values.
        """
        # Unstack the inputs
        _sequences, *_adds, indices = x

        # Apply sequence weights if provided
        if self.context.sequence_weights is not None:
            weights = tf.gather(
                tf.convert_to_tensor(
                    self.context.sequence_weights, dtype=tf.float32
                ),
                indices,
            )
            weighted_y_pred = y_pred * weights
            weighted_y_pred /= tf.reduce_sum(weights, axis=0)
        else:
            weighted_y_pred = y_pred
            weighted_y_pred /= tf.cast(tf.shape(y_pred)[0], tf.float32)

        # Reduce over the sequence dimension
        weighted_loglik = tf.reduce_sum(weighted_y_pred, axis=0) # (H,)

        return weighted_loglik

    def log_prior(self) -> tf.Tensor:
        """ Computes the logarithmic prior value of each underlying model.

        Returns:
            log_prior: Tensor of shape (num_models,) with the log prior values.
        """
        if self.context.sequence_weights is not None:
            S = self.context.sequence_weights.sum()
        else:
            S = self.context.num_seq
        log_prior = self.phmm_layer.prior_scores()
        if S > 0:
            log_prior /= tf.cast(S, tf.float32)
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
    def metrics(self):
        return [self.loss_tracker, self.loglik_tracker, self.prior_tracker]

    def predict(
        self,
        data: SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
        indices: np.ndarray | None = None,
        models: list[int] | None = None,
        bucket_boundaries: Sequence[int | float] | None = None,
        bucket_batch_sizes: Sequence[int] | None = None,
        reduce: bool = False,
    ) -> np.ndarray:
        """
        Computes predictions for all sequences specified by indices in data.
        Buckets sequences by length for efficiency and determines appropriate
        batch sizes from the `TrainingConfiguration` of the model.
        Specify a call mode before running this method, e.g. `viterbi_mode()`.

        Args:
            data: Dataset or tuple of Dataset(s) containing
                the sequences to
                train on. When multiple datasets are provided, they are
                required to have the same order (i.e. index i in each
                dataset corresponds to the same sequence) and the HMM must
                use a fitting emitter for each dataset. The first dataset
                will be used to query metadata such as the number of sequences
                and sequence lengths, although these metrics should be
                consistent across datasets.
            indices: Array of sequence indices to predict on
            models: List of model indices to use for prediction
            bucket_boundaries: Sequence length boundaries for bucketing. If None,
                uses default boundaries [200, 520, 700, 850, 1200, 2000, 4000, inf].
            bucket_batch_sizes: Batch sizes for each bucket. If None, uses the
                adaptive batch size function from the batch generator.
            reduce: If True and in posterior mode, reduces over sequences and
                positions to return state expectations instead of full posteriors.
                This saves memory for large datasets.

        Returns: An array whose shape depends on the call mode and reduce option:
            - Viterbi mode: (num_sequences, length, num_models) with the
                most likely state sequences for each sequence and model.
            - Loglik mode: (num_sequences, num_models) with the loglikelihoods
              for each sequence and model.
            - Posterior mode (reduce=False):
                (num_sequences, length, num_models, num_states)
                with the posterior state distribution per sequence position
                and head. Outputs zeros for padding positions.
            - Posterior mode (reduce=True): (num_models, max_num_states)
                with the expected number of visits per state, averaged over
                sequences.
        """
        data = self._pack_datasets(data, "predict")

        if indices is None:
            indices = np.arange(data[0].num_seq)
        start_time = time.perf_counter()

        # restrict to specified models
        # TODO: revert head_subset later?
        self.phmm_layer.head_subset = models
        if self.context.config.training.use_anc_probs:
            self.anc_probs_layer.head_subset = models

        if models is None:
            _models = list(range(len(self.phmm_layer.lengths)))
        else:
            _models = models

        # Additional setup required before running Viterbi
        # TODO: clean up later
        self.context.batch_gen.configure(data, context=self.context)
        # Don't use static shapes for prediction - we'll use bucketing
        self.context.batch_gen.static_shape_mode = False
        old_crop_long_seqs = self.context.batch_gen.crop_long_seqs
        self.context.batch_gen.crop_long_seqs = math.inf #do not crop sequences

        # Override the number of models from the config if necessary
        if self.phmm_layer.head_subset is not None:
            self.context.batch_gen.num_models = len(self.phmm_layer.head_subset)

        if bucket_boundaries is None or bucket_batch_sizes is None:
            bucket_boundaries, bucket_batch_sizes = make_default_bucket_scheme(
                indices=indices,
                batch_generator=self.context.batch_gen,
                model_lengths=[self.phmm_layer.lengths[m] for m in _models],
                batch_size_impl_factor=self.context._get_impl_factor(True),
            )

        # Create dataset and get number of steps
        ds, steps = make_dataset(
            indices,
            self.context.batch_gen,
            shuffle=False,
            bucket_by_seq_length=True,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
        )

        self._print_predict_header(
            indices, bucket_boundaries, bucket_batch_sizes, steps
        )

        assert steps > 0,\
            "Prediction dataset must have a positive, finite number of steps."

        # Compile to acccount for any changes in head_subset or call mode
        self.compile(total_steps=steps)

        if reduce and self.phmm_layer.is_posterior_mode():
            # Special reduced posterior mode: accumulate state posteriors
            # online to avoid storing full arrays in memory
            Q = max(self.phmm_layer.states[m] for m in _models)
            H = len(_models)
            accumulated_posteriors = np.zeros((H, Q), dtype=np.float32)

            def reduce_batch(batch_data):
                y = self.predict_step(batch_data)
                # Drop indices if present (bucketing)
                y = y[0] if isinstance(y, tuple) else y
                # Sum over sequence positions (axis 1) and batch (axis 0)
                return tf.reduce_sum(y, axis=[0, 1])

            if self.use_jit_compile(steps):
                reduce_batch = tf.function(reduce_batch, jit_compile=True)

            for batch_data in ds.take(steps):
                y = reduce_batch(batch_data)
                accumulated_posteriors += y.numpy()[..., :-1] #drop terminal

            accumulated_posteriors /= len(indices)
            self.context.batch_gen.crop_long_seqs = old_crop_long_seqs
            self._print_predict_timing(
                elapsed_seconds=time.perf_counter() - start_time,
                num_sequences=len(indices),
                steps=steps,
            )
            return accumulated_posteriors

        # Run a custom prediction loop with batching to collect all predictions
        all_predictions = []
        all_indices = []

        def predict_batch(batch_data):
            return self.predict_step(batch_data)

        if self.use_jit_compile(steps):
            predict_batch = tf.function(predict_batch, jit_compile=True)

        for batch_data in ds.take(steps):
            batch_result = predict_batch(batch_data)
            # Standard mode: collect predictions
            if isinstance(batch_result, tuple) and len(batch_result) == 2:
                batch_pred, batch_idx = batch_result
                all_predictions.append(batch_pred.numpy())
                all_indices.append(batch_idx.numpy())
            elif isinstance(batch_result, tf.Tensor):
                all_predictions.append(batch_result.numpy())

        # Handle variable lengths - slice bucket padding and optionally pad
        # Use actual data max length (+1 for terminal), not bucket padding
        max_len = int(max(data[0].seq_lens[indices]) + 1)

        if self.phmm_layer.is_posterior_mode()\
                or self.phmm_layer.is_viterbi_mode():
            # Process predictions: slice if too long, pad if needed
            # (only in posterior/loglik modes)
            processed_predictions = []
            for pred in all_predictions:
                # Slice off bucket padding if prediction is too long
                if pred.shape[1] > max_len:
                    pred = pred[:, :max_len]

                # Pad if needed and in posterior/loglik mode
                if pred.shape[1] < max_len:
                    if self.phmm_layer.is_posterior_mode():
                        pad_value = 0
                    else:
                        pad_value = -1
                    pad_width = [(0, 0)] * len(pred.shape)
                    pad_width[1] = (0, max_len - pred.shape[1])
                    pred = np.pad(pred, pad_width, constant_values=pad_value)
                    if self.phmm_layer.is_viterbi_mode():
                        for i, j in enumerate(_models):
                            q = self.phmm_layer.states[j]
                            pred[...,i][pred[...,i] == -1] = q

                processed_predictions.append(pred)
        else:
            # In loglik mode, no slicing/padding needed
            processed_predictions = all_predictions

        # Concatenate all predictions
        decoded_array = np.concatenate(processed_predictions, axis=0)

        # If bucketing was used, reorder to original sequence order
        if all_indices:
            bucket_indices = np.concatenate(all_indices, axis=0)
            sorted_order = np.argsort(bucket_indices)
            decoded_array = decoded_array[sorted_order]

        # Reset
        self.context.batch_gen.crop_long_seqs = old_crop_long_seqs

        # Replace -1 padding
        if self.phmm_layer.is_posterior_mode():
            decoded_array[decoded_array == -1] = 0
        elif self.phmm_layer.is_viterbi_mode():
            for i,j in enumerate(_models):
                L = self.phmm_layer.lengths[j]
                decoded_array[:,:,i][decoded_array[:,:,i] == -1] = 2*L + 2

        self._print_predict_timing(
            elapsed_seconds=time.perf_counter() - start_time,
            num_sequences=len(indices),
            steps=steps,
        )

        return decoded_array

    @override
    def predict_step(self, data : Any) -> Any:
        """
        Custom prediction step that handles the optional index for bucketed
        datasets.

        Args:
            data: Either ((batch, indices), y) for regular datasets
                or ((batch, indices, j), y) for bucketed datasets where j is the
                original sequence index.

        Returns:
            If bucketed: (predictions, j) to allow reordering
            Otherwise: predictions
        """
        x, _ = data

        # Bucketed dataset, we have reordering index j
        batch, *adds, indices, j = x

        predictions = self((batch, *adds, indices), training=False)

        # Return predictions along with the index for reordering
        return predictions, j

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
            and 'prior' (per-model) when in evaluation mode.
        """
        data = self._pack_datasets(data, "evaluate")

        if indices is None:
            indices = np.arange(data[0].num_seq)

        self.loglik_mode()

        # restrict to specified models
        # TODO: revert head_subset later?
        self.phmm_layer.head_subset = models
        if self.context.config.training.use_anc_probs:
            self.anc_probs_layer.head_subset = models

        if models is None:
            _models = list(range(len(self.phmm_layer.lengths)))
        else:
            _models = models

        # Additional setup required before running Viterbi
        # TODO: clean up later
        self.context.batch_gen.configure(data, context=self.context)
        # Don't use static shapes for prediction - we'll use bucketing
        self.context.batch_gen.static_shape_mode = False
        old_crop_long_seqs = self.context.batch_gen.crop_long_seqs
        self.context.batch_gen.crop_long_seqs = math.inf #do not crop sequences

        # Override the number of models from the config if necessary
        if self.phmm_layer.head_subset is not None:
            self.context.batch_gen.num_models = len(self.phmm_layer.head_subset)

        # Create dataset and get number of steps
        bucket_boundaries, bucket_batch_sizes = make_default_bucket_scheme(
            indices=indices,
            batch_generator=self.context.batch_gen,
            model_lengths=[self.phmm_layer.lengths[m] for m in _models],
            batch_size_impl_factor=self.context._get_impl_factor(True),
        )

        ds, steps = make_dataset(
            indices,
            self.context.batch_gen,
            shuffle=False,
            bucket_by_seq_length=True,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
        )

        self._print_predict_header(
            indices, bucket_boundaries, bucket_batch_sizes, steps
        )

        # Compile to acccount for any changes in head_subset or call mode
        self.compile(total_steps=steps)

        # Enable eval mode and reset per-model trackers
        self._eval_mode = True
        for tracker in self.loglik_per_model_trackers:
            tracker.reset_state()
        for tracker in self.prior_per_model_trackers:
            tracker.reset_state()

        # Use None for infinite steps (-1), otherwise use the computed steps
        steps_param = None if steps == -1 else steps
        result = super().evaluate(
            ds, steps=steps_param, verbose=self.get_verbosity()
        )

        # Disable eval mode
        self._eval_mode = False

        # Get per-model results from trackers
        loglik_per_model = np.array([
            tracker.result().numpy() for tracker in self.loglik_per_model_trackers
        ])
        prior_per_model = np.array([
            tracker.result().numpy() for tracker in self.prior_per_model_trackers
        ])

        # Get scalar loss from result
        if isinstance(result, list):
            total_loss = result[0]
        else:
            total_loss = result

        # Reset
        self.context.batch_gen.crop_long_seqs = old_crop_long_seqs

        return {
            'loss': np.asarray(total_loss),
            'loglik': loglik_per_model,
            'prior': prior_per_model,
        }

    @override
    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        """
        Custom test step that handles the optional index for bucketed datasets.

        Args:
            data: Either ((batch, indices), y) for regular datasets
                or ((batch, indices, j), y) for bucketed datasets where j is the
                original sequence index.

        Returns:
            Dictionary of metric results
        """
        x, y = data

        # Check if we have the bucketing index and strip it
        #if isinstance(x, tuple) and len(x) == 3:
        # Bucketed dataset: (batch, indices, j) - extract (batch, indices)
        batch, *adds, indices, _j = x
        x = (batch, *adds, indices)

        # Compute predictions
        y_pred = self(x, training=False)

        # Compute loss (updates metrics internally)
        self.compute_loss(x, y, y_pred)

        # Return metrics
        return {m.name: m.result() for m in self.metrics}

    def estimate_loglik(
        self,
        data: SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
        max_seq: int = 200000,
        reduce: bool = True,
        models: list[int] | None = None
    ) -> np.ndarray:
        """ Computes the logarithmic likelihood for each underlying model.

        Args:
            max_seq: Threshold for the number of sequences used to compute the
                loglik. If the dataset has more sequences, a random subset is
                drawn.
            reduce: If true, the loglik will be averaged over the number of
                sequences.
            models: List of model indices for which the loglik should be
                computed.

        Returns:
            loglik: Logarithmic likelihoods. If reduce is true, the shape is
                (num_models,), otherwise (num_sequences, num_models).
        """
        if isinstance(data, SequenceDataset):
            n = data.num_seq
        else:
            n = data[0].num_seq
        if n > max_seq:
            # estimate the ll only on a subset for efficiency
            indices = np.arange(n)
            np.random.shuffle(indices)
            indices = indices[:max_seq]
            indices = np.sort(indices)
        else:
            indices = np.arange(n)
        if reduce:
            return self.evaluate(data, indices=indices, models=models)["loglik"]
        else:
            self.loglik_mode()
            self.compile(total_steps=len(indices))
            return self.predict(data, indices=indices, models=models)

    def _pack_datasets(
        self,
        data: SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
        method_name: str,
    ) -> tuple[SequenceDataset, *tuple[Dataset, ...]]:
        if isinstance(data, SequenceDataset):
            return (data,)
        if len(data) == 0:
            raise ValueError(f"Model.{method_name} requires at least one dataset.")
        if not isinstance(data[0], SequenceDataset):
            raise ValueError(
                f"The first dataset in the tuple passed to Model.{method_name} "
                "must be a SequenceDataset."
            )
        return data

    def compute_null_model_log_probs(
        self,
        data: Dataset,
        background_dist: np.ndarray | None = None,
        transition_prob: float | None = None
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
                    from the emitter. Must be a flat array of size alphabet_size.
                transition_prob: Optional self-loop transition probability.
                    If None, uses M/(M+1) where M is the mean sequence length.
                    Must be a probability value in [0, 1].

            Returns:
                log_probs: Array of log probabilities for each sequence.
        """
        # Prepare the data
        n = data.num_seq

        # Configure batch generator
        self.context.batch_gen.configure(data, context=self.context)

        null_indices = np.arange(n)
        bucket_boundaries, bucket_batch_sizes = make_default_bucket_scheme(
            indices=null_indices,
            batch_generator=self.context.batch_gen,
            model_lengths=self.phmm_layer.lengths,
            # effectively doubles the batch size compared to training
            batch_size_impl_factor=0.5,
        )

        ds, _ = make_dataset(
            null_indices,
            self.context.batch_gen,
            shuffle=False,
            bucket_by_seq_length=True,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
        )

        # Prepare the background frequencies
        if background_dist is None:
            # Use prior background distribution
            assert self.phmm_layer.hmm.emitter[0].prior,\
                    "Emitter needs a Dirichlet prior for null model computation."
            dirichlet_alpha = self.phmm_layer.hmm.emitter[0].prior.matrix().numpy()
            dirichlet_alpha = dirichlet_alpha[0,0] # Shared, pick any head and state
            _background_dist = dirichlet_alpha / np.sum(dirichlet_alpha)

            # Append ad-hoc probability for non-standard amino acids
            _background_dist = np.append(_background_dist, [1e-4]*3)
            _background_dist /= np.sum(_background_dist) #normalize  # shape (24,)
        else:
            _background_dist = background_dist

        # Add the terminal symbol emissions
        _background_dist = np.append(_background_dist, [1.0])
        # Log transition probabilities
        _background_dist = np.log(_background_dist + 1e-10)

        # Compute emission log probs
        log_probs = np.zeros((n,))
        for batch_data, _ in ds:
            # Handle bucketed datasets
            if isinstance(batch_data, tuple) and len(batch_data) == 3:
                x, _, batch_idx = batch_data
            else:
                x, _ = batch_data
                # No bucket indices available, skip this batch
                continue

            x_np = x.numpy()[:, :, 0]  # (batch, seq_length)
            em = np.sum(_background_dist[x_np], axis=1)
            log_probs[batch_idx.numpy()] = em

        # Add transition log probs based on the target sequence lengths
        # We'll assume a geometric distribution with the expected length
        # equal to the length of each target sequence
        L = data.seq_lens
        if transition_prob is None:
            M = np.mean(L)
            trans_scores = (L - 1) * (np.log(M) - np.log(M+1))
        else:
            trans_scores = (L - 1) * np.log(transition_prob)
        log_probs += trans_scores

        return log_probs

    def estimate_AIC(
        self,
        data: SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
        max_seq: int = 200000,
        loglik: np.ndarray | None = None
    ) -> np.ndarray:
        """ Computes the Akaike information criterion for each underlying model.

        Args:
            data: SequenceDataset containing the sequences to evaluate.
            max_seq: Threshold for the number of sequences used to compute the
                loglik. If the dataset has more sequences, a random subset is
                drawn.
            loglik: This argument can be set if the loglik was computed before
                via estimate_loglik to avoid overhead. If None, the loglik will
                be computed internally.

        Returns:
            aic: Array of AIC values for each model.
        """
        if isinstance(data, SequenceDataset):
            data = (data,)
        if loglik is None:
            loglik = self.estimate_loglik(data, max_seq, reduce=True)
        num_param = 34 * np.array(self.phmm_layer.lengths) + 25
        aic = -2 * loglik * data[0].num_seq + 2 * num_param
        return aic

    def compute_consensus_score(self) -> tf.Tensor:
        """ Computes a consensus score that rates how plausible each model is
            with respect to all other models.
            (Relevant for users not using the default emitter: Uses the
            make_B_amino method of the first emitter.)

        Returns:
            consensus_score: Tensor of shape (num_models,) with consensus scores.
        """
        num_models = self.context.config.training.num_model
        model_lengths = self.phmm_layer.lengths
        alphabet_size = self.context.config.hmm.alphabet_size

        # compute the match sequence of all models padded with terminal symbols
        match_seqs = np.zeros(
            (num_models, max(model_lengths)+1, alphabet_size + 1)
        )
        match_seqs[:,:,-1] = 1  # initialize with terminal symbols
        emitter = self.phmm_layer.hmm.emitter[0]
        for i, L in enumerate(model_lengths):
            match_seqs[i, :L] = emitter.matrix()[i, 1:L+1]
        # we need to tile the match sequences over the batch dimension because
        # each model should see all other models
        match_seqs = tf.stack([match_seqs] * num_models, axis=1)
        # rate the match seqs with respect to the models and cancel out
        # self-rating

        # TODO this does not work with self.model which expects indices rather
        # than distributions
        # this is a workaround, but will break with a user defined encoder model
        # we skip the anc probs for simplicity as this would require fitting
        # evolutionary times
        consensus_logliks = self.phmm_layer.hmm.likelihood_log(
            match_seqs[:, :, :, :-1],  # Remove terminal symbol for emissions
            1 - match_seqs[:, :, :, -1:]  # Padding mask
        )

        consensus_logliks *= 1 - tf.eye(num_models)
        # Axis 1 means we reduce over the batch dimension rather than the model
        # dimension,
        # so output i will be the mean loglik if we input all other match
        # sequenes to model i.
        # Using axis 0 here is not the same!
        # Consider the case that all models have the same match sequence but
        # one model allows many deletions or insertions.
        # This model is the outlier and should clearly have the lowest score.
        # With axis=0, the likelihood of the outlier model under all other
        # models is high and the scores of the other models will have a penalty
        # since their match sequences are fed into the outlier model.
        # With axis=1, the scores of all models will be high except for the
        # outlier model, which has a strong penalty as it rates all other
        # match sequences involving the deletion/insertion probabilities.
        consensus_score = tf.reduce_mean(consensus_logliks, axis=1)
        return consensus_score

    def get_batch_size(self, data:SequenceDataset) -> int:
        if callable(self.context.batch_size):
            return self.context.batch_size(data)
        else:
            return self.context.batch_size

    def get_num_epochs(self, iteration: int) -> int:
        """
        Determine the number of epochs for the current training iteration.

        Args:
            iteration: Current iteration number in the training loop.

        Returns:
            Number of epochs to train for this iteration.
        """
        last_iteration = (
            iteration == self.context.config.training.max_iterations - 1
        )
        epochs = self.context.config.training.epochs[
            0 if iteration==0 else 1 if not last_iteration else 2
        ]
        return epochs

    def get_num_steps(
        self, num_sequences: int, batch_size: int, min_steps: int = 5
    ) -> int:
        """
        Determine the number of steps per epoch based on the number of sequences
        and batch size.

        Args:
            num_sequences: Total number of sequences to train on.
            batch_size: Number of sequences per batch.

        Returns:
            Number of steps per epoch.
        """
        if num_sequences == 0 or batch_size == 0:
            return 0
        steps = int(100*np.sqrt(num_sequences)/batch_size)
        return min(max(min_steps, steps), 500)

    def get_train_callbacks(self) -> list[tf.keras.callbacks.Callback]:
        """
        Create and return a list of standard training callbacks
        (early stopping, terminate on NaN).

        Returns:
            List of Keras callbacks for training.
        """
        terminate_on_nan = TerminateOnNaNWithCheckpoint(
            self, self.context.config.input_output.work_dir
        )
        early_stopping = tf.keras.callbacks.EarlyStopping("loss", patience=1)
        return [terminate_on_nan, early_stopping]

    def use_jit_compile(self, total_steps: int | None = None) -> bool:
        """
        Determine whether to use JIT compilation for training.

        Args:
            total_steps: The total number of steps the model will be called for
                (optional). If provided, it is used to decide if JIT should be
                enabled based on the threshold.

        Returns:
            True if JIT compilation should be used, False otherwise.
        """
        jit_compile = self.context.config.advanced.jit_compile
        if total_steps is not None:
            # jit compilation becomes very slow for long HMMs
            # (say > 450 matches)
            # make sure we only enable it if we will be running long enough to
            # benefit from it
            jit_compile = jit_compile and total_steps >= 20
            if max(self.context.model_lengths) > 450:
                jit_compile = jit_compile and total_steps >= 100
        return jit_compile

    def get_verbosity(self) -> Literal[0, 2]:
        """
        Determine the verbosity level for training output.

        Returns:
            Verbosity level (0 for silent, 2 for verbose).
        """
        return 2 if self.context.config.input_output.verbose else 0

    def get_config(self):
        """
        Returns the config of the model.

        A model config is a Python dictionary (serializable) containing the
        configuration of the model. The same config can be used to reinstantiate
        the model via `from_config()`.
        """
        # Get the base class config which includes 'trainable', 'dtype', etc.
        base_config = super().get_config()
        base_config["learnmsa_context"] = self.context
        return base_config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """
        Creates a model from its config.

        This method is the reverse of `get_config`, capable of instantiating the
        same model from the config dictionary.

        Note: Since LearnMSAModel requires a LearnMSAContext object which cannot
        be serialized, this method creates a placeholder instance during
        deserialization. The actual model structure and weights are restored by
        Keras from the saved file.

        Args:
            config: A Python dictionary, typically the output of get_config.
            custom_objects: Optional dictionary mapping names to custom classes
                          or functions.

        Returns:
            A LearnMSAModel instance (in deserialization mode).
        """
        # Remove our custom marker
        config_copy = config.copy()
        learnmsa_context_dict = config_copy.pop("learnmsa_context")
        context = LearnMSAContext.from_config(learnmsa_context_dict)

        # Create an instance with context=None to signal deserialization mode
        # Keras will restore the model structure and weights from the saved file
        return cls(context, **config_copy)

    def _print_train_header(
        self, indices: np.ndarray, batch_size: int, data: Dataset
    ) -> None:
        if self.context.config.input_output.verbose:
            print(
                "Fitting models of lengths",
                self.context.model_lengths, "on", indices.shape[0], "sequences"
            )
            print(
                "Batch size=", batch_size,
                "Learning rate=", self.context.config.training.learning_rate
            )
            if self.context.sequence_weights is not None:
                io = self.context.config.input_output
                input_path = Path(io.input_file)
                if input_path.name:
                    weight_path = Path(io.work_dir) /\
                        input_path.with_suffix(".weights").name
                else:
                    weight_path = Path(io.work_dir) / "sequences.weights"
                print("Using sequence weights and writing them to", weight_path)
                write_sequence_weights(
                    data, self.context.sequence_weights, str(weight_path)
                )
            else:
                print("Don't use sequence weights")
            if int(self.context.batch_gen.crop_long_seqs) < math.inf:
                num_cropped = np.sum(
                    data.seq_lens[indices] >\
                        self.context.batch_gen.crop_long_seqs
                    )
                if num_cropped > 0:
                    print(
                        f"{num_cropped} sequences are longer than "
                        f"{self.context.batch_gen.crop_long_seqs} and will be "\
                        "cropped for training.\nTo disable cropping, use "\
                        "--crop disable. To change the cropping limit to X, "\
                        "use --crop X."
                    )
            if self.phmm_layer.use_language_model:
                print("Protein language model support is enabled")
            num_gpu = len([x.name for x in tf.config.list_logical_devices() if x.device_type == 'GPU'])
            if num_gpu == 0:
                print("Using CPU")
            else:
                print("Using GPU")

    def _print_predict_header(
        self, indices: np.ndarray,
        bucket_boundaries: Sequence[int | float],
        bucket_batch_sizes: Sequence[int],
        steps: int,
    ) -> None:
        if self.context.config.input_output.verbose:
            print(
                "Predicting on", indices.shape[0], "sequences with bucket ",
                "boundaries", bucket_boundaries, "and batch sizes",
                bucket_batch_sizes[:-1], "for", steps, "steps"
            )

    def _print_predict_timing(
        self,
        elapsed_seconds: float,
        num_sequences: int,
        steps: int,
    ) -> None:
        if self.context.config.input_output.verbose:
            if elapsed_seconds > 0.0:
                seq_per_s = num_sequences / elapsed_seconds
                print(
                    f"Prediction finished in {elapsed_seconds:.3f}s "
                    f"({seq_per_s:.2f} seq/s, {steps} steps)"
                )
            else:
                print(
                    f"Prediction finished in {elapsed_seconds:.3f}s "
                    f"({steps} steps)"
                )

    def _check_training_complete(
        self,
        history: tf.keras.callbacks.History
    ) -> None:
        # Check if the last reported loss is NaN and terminate if so
        if history.history and "loss" in history.history:
            final_loss = history.history['loss'][-1]
            if math.isnan(final_loss):
                error_msg = "Training terminated: Final loss is NaN."\
                    f" Loss history: {history.history['loss']}"
                raise ValueError(error_msg)

        if self.context.config.input_output.verbose:
            print("Fitted model successfully.")


# Register custom objects for serialization
tf.keras.utils.get_custom_objects()["LearnMSAModel"] = LearnMSAModel
