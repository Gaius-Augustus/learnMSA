import math
from typing import Any, Literal, override

import numpy as np
import tensorflow as tf

import learnMSA.msa_hmm.Viterbi as viterbi
from learnMSA.hmm.tf.layer import PHMMLayer
from learnMSA.msa_hmm.AncProbsLayer import AncProbsLayer
from learnMSA.msa_hmm.learnmsa_context import LearnMSAContext
from learnMSA.msa_hmm.MsaHmmCell import MsaHmmCell
from learnMSA.msa_hmm.MsaHmmLayer import MsaHmmLayer
from learnMSA.msa_hmm.posterior import get_state_expectations
from learnMSA.msa_hmm.training import make_dataset
from learnMSA.util.sequence_dataset import SequenceDataset


class LearnMSAModel(tf.keras.Model):
    """
    Main LearnMSA model for training and decoding.
    """

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
        if train_cfg.use_anc_probs:
            if len(context.encoder_initializer) > 3:
                matrix_rate_init = context.encoder_initializer[3]
            else:
                matrix_rate_init = None
            self.anc_probs_layer = AncProbsLayer(
                train_cfg.num_model,
                context.num_seq,
                train_cfg.num_rate_matrices,
                equilibrium_init=context.encoder_initializer[2],
                rate_init=context.encoder_initializer[0],
                exchangeability_init=context.encoder_initializer[1],
                trainable_rate_matrices=train_cfg.trainable_rate_matrices,
                trainable_distances=train_cfg.trainable_distances,
                per_matrix_rate=train_cfg.per_matrix_rate,
                matrix_rate_init=matrix_rate_init,
                matrix_rate_l2=train_cfg.matrix_rate_l2,
                shared_matrix=train_cfg.shared_rate_matrix,
                equilibrium_sample=train_cfg.equilibrium_sample,
                transposed=train_cfg.transposed,
                clusters=context.clusters
            )

        self.phmm_layer = PHMMLayer(
            context.model_lengths,
            config = context.config.hmm,
            prior_config = context.config.hmm_prior,
            plm_config = context.config.language_model,
            use_prior = context.config.training.use_prior,
        )

        # Metrics trackers
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.loglik_tracker = tf.keras.metrics.Mean(name='loglik')
        self.prior_tracker = tf.keras.metrics.Mean(name='prior')

    def loglik_mode(self) -> None:
        """Makes the model return log-likelihoods.
        """
        self.phmm_layer.loglik_mode()

    def viterbi_mode(self) -> None:
        """Makes the model return Viterbi paths.
        """
        self.phmm_layer.viterbi_mode()

    def posterior_mode(self) -> None:
        """Makes the model return state posterior probabilities.
        """
        self.phmm_layer.posterior_mode()

    def call(
        self,
        inputs: tuple[tf.Tensor | np.ndarray, ...],
        training: bool|None=None,
    ) -> tuple[tf.Tensor, ...]:
        """
        Forward pass of the model.

        Args:
            inputs: Tuple of (sequences, indices) where:
                   - sequences: shape (batch, num_models, seq_length)
                   - indices: shape (batch, num_models)
                   - embeddings: shape (batch, num_models, seq_length, dim)
                        If in language model mode.
            training: Boolean indicating training mode.

        Returns:
            Tuple of outputs depending on use_prior:
            - Without prior: (loglik, aggregated_loglik)
            - With prior: (loglik, aggregated_loglik, prior, aux_loss)
        """
        # Pass through encoder layers
        forward_seq = self.encode_batch(inputs, training=training)

        # transpose back
        # TODO: clean up later, no transpose should be necessary
        forward_seq = tf.transpose(forward_seq, [1, 2, 0, 3])
        if self.context.config.language_model.use_language_model:
            embeddings = inputs[-1]
            embeddings = tf.transpose(embeddings, [0, 2, 1, 3])
        else:
            embeddings = None

        padding = 1 - forward_seq[:, :, :, -1:]
        forward_seq = forward_seq[:, :, :, :-1]

        output = self.phmm_layer(
            forward_seq, embeddings=embeddings, padding=padding
        )

        if self.phmm_layer.is_loglik_mode():
            output = tf.squeeze(output, axis=-1)

        return output

    def encode_batch(
        self,
        inputs: tuple[tf.Tensor | np.ndarray, ...],
        training: bool|None=None,
    ) -> tf.Tensor:
        """
        Encodes a batch of sequences with the ancestor probabilities layer,
        i.e. computes the HMM inputs.

        Args:
            inputs: Tuple of (sequences, indices) where:
                   - sequences: shape (batch, num_models, seq_length)
                   - indices: shape (batch, num_models)
                   - embeddings: shape (batch, num_models, seq_length, dim)
                        If in language model mode. (not used)
            training: Boolean indicating training mode.

        Returns:
            A tensor with the ancestral probabilities of the input sequences
            as inputs to the pHMM layer.
        """
        if self.context.config.language_model.use_language_model:
            sequences, indices, embeddings = inputs
        else:
            sequences, indices = inputs
            embeddings = None

        # Broadcast in the number of heads if necessary
        n = self.context.config.training.num_model
        if sequences.shape[1] == 1 and n > 1:
            sequences = tf.tile(sequences, [1, n, 1])
            indices = tf.tile(indices, [1, n])
            if embeddings is not None:
                embeddings = tf.tile(embeddings, [1, n, 1, 1])

        # Transpose: (batch, num_models, L) -> (num_models, batch, L)
        # In the input pipeline, we need the batch dimension to come first to
        # make multi GPU work. We transpose here, because all learnMSA layers
        # require the model dimension to come first
        # TODO: clean this up; hidten does not use heads first order anymore;
        # needs merging of tree branch first
        transposed_sequences = tf.transpose(sequences, [1, 0, 2])
        transposed_indices = tf.transpose(indices, [1, 0])

        # Pass through encoder layers
        encoded_seq = transposed_sequences
        if self.context.config.training.use_anc_probs:
            encoded_seq = self.anc_probs_layer(
                encoded_seq, rate_indices=transposed_indices, training=training # type: ignore
            )
        else:
            encoded_seq = tf.one_hot(
                encoded_seq,
                depth=self.context.config.hmm.alphabet_size+1, # including padding
                dtype=self.phmm_layer.dtype
            )
        return encoded_seq

    @override
    def build(self, input_shapes: tuple[tuple[int | None, ...], ...]):
        """
        Build the model based on the input shape.

        Args:
            input_shape: Shape of the input data.
        """
        batch_size = input_shapes[0][0]
        s = self.context.config.hmm.alphabet_size
        n = self.context.config.training.num_model
        seq_shape_t = (n, batch_size, None)
        ind_shape_t = (n, batch_size)
        if self.context.config.training.use_anc_probs:
            self.anc_probs_layer.build([seq_shape_t, ind_shape_t])

        # Build the pHMM layer
        if self.context.config.language_model.use_language_model:
            emb_dim = self.context.config.language_model.scoring_model_dim
            self.phmm_layer.build(
                input_shape=(
                    (batch_size, None, n, s),
                    (batch_size, None, n, emb_dim),
                    (batch_size, None, n, 1),
                )
            )
        else:
            self.phmm_layer.build(
                input_shape=(
                    (batch_size, None, n, s),
                    (batch_size, None, n, 1),
                )
            )

    @override
    def compile(self) -> None:
        """
        Compile the model with Adam optimizer.

        This method compiles the LearnMSA model using the Adam optimizer
        with the learning rate specified in the configuration.
        """
        optimizer = tf.keras.optimizers.Adam(
            self.context.config.training.learning_rate
        )
        super().compile(
            optimizer=optimizer,
            jit_compile=self.context.config.advanced.jit_compile,
        )

    @override
    def fit(
        self,
        data: SequenceDataset,
        indices: np.ndarray,
        iteration: int,
        batch_size: int,
        callbacks: list[tf.keras.callbacks.Callback]=[],
    ) -> tf.keras.callbacks.History:
        """
        Fit the LearnMSA model on the specified sequences.

        Args:
            data: SequenceDataset containing the sequences to train on
            indices: Array of sequence indices to train on
            iteration: Current iteration number in the training loop
            batch_size: Number of sequences per batch
            callbacks: List of Keras callbacks to use during training

        Returns:
            Training history object containing loss and metrics
        """
        self.phmm_layer.loglik_mode()
        self.context.batch_gen.configure(data, self.context)
        self._print_train_header(indices, batch_size, data)
        dataset = make_dataset(
            indices,
            self.context.batch_gen,
            batch_size,
            shuffle=True
        )
        history = super().fit(
            dataset,
            epochs=self.get_num_epochs(iteration),
            steps_per_epoch=self.get_num_steps(indices.shape[0], batch_size),
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
        """
        # Unstack the inputs
        if self.context.config.language_model.use_language_model:
            _sequences, indices, _embeddings = x
        else:
            _sequences, indices = x

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
            total_weight_sum = self.context.sequence_weights.sum()
        else:
            weighted_y_pred = y_pred
            total_weight_sum = self.context.num_seq

        # Reduce over the sequence dimension
        weighted_loglik = tf.reduce_sum(weighted_y_pred, axis=0) # (H,)
        weighted_loglik_mean = tf.reduce_mean(weighted_loglik)

        # Compute the full loss
        log_prior = self.phmm_layer.prior_scores()
        log_prior /= tf.cast(total_weight_sum, tf.float32) # Scale prior
        log_prior_mean = tf.reduce_mean(log_prior)

        loss = - weighted_loglik_mean - log_prior_mean

        self.loss_tracker.update_state(loss)
        self.loglik_tracker.update_state(-weighted_loglik_mean)
        self.prior_tracker.update_state(-log_prior_mean)

        return loss

    def reset_metrics(self) -> None:
        """Reset all metric trackers."""
        self.loss_tracker.reset_state()
        self.loglik_tracker.reset_state()
        self.prior_tracker.reset_state()
        self.aux_loss_tracker.reset_state()

    @property
    def metrics(self):
        return [self.loss_tracker, self.loglik_tracker, self.prior_tracker]

    def decode(
        self,
        data: SequenceDataset,
        indices: np.ndarray,
        batch_size: int,
        models: list[int],
        non_homogeneous_mask_func=None,
    ) -> np.ndarray:
        self.phmm_layer.viterbi_mode()
        assert non_homogeneous_mask_func is None, \
            "Non-homogeneous decoding not implemented yet."
        data = make_dataset(
            sorted_indices[:,0],
            batch_generator,
            batch_size,
            shuffle=False,
            bucket_by_seq_length=True,
            model_lengths=cell.length
        )
        return self.evaluate(data)

    def posterior(
        self,
        data: SequenceDataset,
        indices: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        self.encode_only = True
        expected_states = get_state_expectations(
            data,
            self.context.batch_gen,
            indices,
            batch_size,
            self.msa_hmm_layer,
            self,
            with_plm=self.context.config.language_model.use_language_model,
            plm_dim=self.context.scoring_model_config.dim
        ).numpy()
        self.encode_only = False
        return expected_states

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

    def get_num_steps(self, num_sequences: int, batch_size: int) -> int:
        """
        Determine the number of steps per epoch based on the number of sequences
        and batch size.

        Args:
            num_sequences: Total number of sequences to train on.
            batch_size: Number of sequences per batch.

        Returns:
            Number of steps per epoch.
        """
        return min(max(10, int(100*np.sqrt(num_sequences)/batch_size)), 500)

    def get_train_callbacks(self) -> list[tf.keras.callbacks.Callback]:
        """
        Create and return a list of standard training callbacks
        (early stopping, terminate on NaN).

        Returns:
            List of Keras callbacks for training.
        """
        terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()
        early_stopping = tf.keras.callbacks.EarlyStopping("loss", patience=1)
        return [terminate_on_nan, early_stopping]

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
        self, indices: np.ndarray, batch_size: int, data: SequenceDataset
    ) -> None:
        if self.context.config.input_output.verbose:
            print(
                "Fitting models of lengths",
                self.context.model_lengths, "on", indices.shape[0], "sequences."
            )
            print(
                "Batch size=", batch_size,
                "Learning rate=", self.context.config.training.learning_rate
            )
            if self.context.sequence_weights is not None:
                print("Using sequence weights ", self.context.sequence_weights, ".")
            else:
                print("Don't use sequence weights.")
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
        num_gpu = len([x.name for x in tf.config.list_logical_devices() if x.device_type == 'GPU'])
        if self.context.config.input_output.verbose:
            if num_gpu == 0:
                print("Using CPU.")
            else:
                print("Using GPU.")

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