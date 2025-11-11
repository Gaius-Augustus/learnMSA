import math
from typing import override

import numpy as np
import tensorflow as tf

import learnMSA.msa_hmm.Viterbi as viterbi
from learnMSA.msa_hmm.AncProbsLayer import AncProbsLayer
from learnMSA.msa_hmm.learnmsa_context import LearnMSAContext
from learnMSA.msa_hmm.MsaHmmCell import MsaHmmCell
from learnMSA.msa_hmm.MsaHmmLayer import MsaHmmLayer
from learnMSA.msa_hmm.posterior import get_state_expectations
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
from learnMSA.msa_hmm.training import make_dataset


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
            context (LearnMSAContext): The context containing model configuration.
                Can be None only during deserialization.
            **kwargs: Additional keyword arguments for the base Model class
                (e.g., trainable, dtype, name). These are used during deserialization.
        """
        # Filter out base Model kwargs before calling super().__init__()
        super().__init__(**kwargs)

        self.context = context
        train_cfg = context.config.training

        self.encoder_layers = []

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
            self.encoder_layers.append(self.anc_probs_layer)

        # Create the HMM cell and -layer
        self.msa_hmm_cell = MsaHmmCell(
            context.model_lengths,
            dim = 24 * train_cfg.num_rate_matrices,
            emitter = context.emitter,
            transitioner = context.transitioner
        )
        self.msa_hmm_layer = MsaHmmLayer(
            self.msa_hmm_cell,
            num_seqs=context.effective_num_seq,
            use_prior=train_cfg.use_prior,
            sequence_weights=context.sequence_weights,
            dtype=tf.float32
        )

        # Metrics trackers
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.loglik_tracker = tf.keras.metrics.Mean(name='loglik')
        self.prior_tracker = tf.keras.metrics.Mean(name='prior')
        self.aux_loss_tracker = tf.keras.metrics.Mean(name='aux_loss')
        self.use_prior = False

        self.encode_only = False

    def call(self, inputs, training=None):
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
        if self.context.config.language_model.use_language_model:
            sequences, indices, embeddings = inputs
        else:
            sequences, indices = inputs
            embeddings = None

        # Transpose: (batch, num_models, L) -> (num_models, batch, L)
        # In the input pipeline, we need the batch dimension to come first to
        # make multi GPU work we transpose here, because all learnMSA layers
        # require the model dimension to come first
        transposed_sequences = tf.transpose(sequences, [1, 0, 2])
        transposed_indices = tf.transpose(indices, [1, 0])
        if embeddings is not None:
            transposed_embeddings = tf.transpose(embeddings, [1, 0, 2, 3])

        # Pass through encoder layers
        forward_seq = transposed_sequences
        for layer in self.encoder_layers:
            forward_seq = layer(forward_seq, transposed_indices, training=training)

        if embeddings is not None:
            forward_seq = tf.concat([forward_seq, transposed_embeddings], -1)

        if self.encode_only:
            return forward_seq

        # Pass through MSA HMM layer
        if self.msa_hmm_layer.use_prior:
            loglik, aggregated_loglik, prior, aux_loss = self.msa_hmm_layer(
                forward_seq, transposed_indices, training=training
            )
            # Transpose loglik back: (num_models, batch) -> (batch, num_models)
            loglik = tf.transpose(loglik, [1, 0])
            return loglik, aggregated_loglik, prior, aux_loss
        else:
            loglik, aggregated_loglik = self.msa_hmm_layer(
                forward_seq, transposed_indices, training=training
            )
            # Transpose loglik back: (num_models, batch) -> (batch, num_models)
            loglik = tf.transpose(loglik, [1, 0])
            return loglik, aggregated_loglik

    @override
    def build(self, batch_size: int | None = None):  # type: ignore[override]
        """
        Build the model based on the input shape.

        Args:
            input_shape: Shape of the input data.
        """
        n = self.context.config.training.num_model
        seq_shape_t = (n, batch_size, None)
        ind_shape_t = (n, batch_size)
        if self.context.config.language_model.use_language_model:
            msa_hmm_shape_t = (n, batch_size, None, 23 + self.context.scoring_model_config.dim+1)
        else:
            msa_hmm_shape_t = (n, batch_size, None, 23)
        if self.context.config.training.use_anc_probs:
            self.anc_probs_layer.build([seq_shape_t, ind_shape_t])
        self.msa_hmm_layer.build(msa_hmm_shape_t)

    @override
    def compile(self) -> None:  # type: ignore[override]
        """
        Compile the model with Adam optimizer.

        This method compiles the LearnMSA model using the Adam optimizer
        with the learning rate specified in the configuration.
        """
        optimizer = tf.keras.optimizers.Adam(self.context.config.training.learning_rate)
        super().compile(optimizer=optimizer, jit_compile=False)

    @override
    def fit(  # type: ignore[override]
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

        tf.keras.backend.clear_session() #frees occupied memory
        tf.get_logger().setLevel('ERROR')

        self.context.batch_gen.configure(data, self.context)
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

        steps = min(max(10, int(100*np.sqrt(indices.shape[0])/batch_size)), 500)
        dataset = make_dataset(
            indices,
            self.context.batch_gen,
            batch_size,
            shuffle=True
        )
        terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()
        early_stopping = tf.keras.callbacks.EarlyStopping("loss", patience=1)
        callbacks += [terminate_on_nan, early_stopping]

        # Find the number of epochs for this iteration
        last_iteration = (
            iteration == self.context.config.training.max_iterations - 1
        )
        epochs = self.context.config.training.epochs[
            0 if iteration==0 else 1 if not last_iteration else 2
        ]
        verbose_level: int = 2 if self.context.config.input_output.verbose else 0
        history = super().fit(
            dataset,
            epochs=epochs,
            steps_per_epoch=steps,
            callbacks=callbacks,
            verbose=verbose_level
        )

        # Check if the last reported loss is NaN and terminate if so
        if history.history and "loss" in history.history:
            final_loss = history.history['loss'][-1]
            if math.isnan(final_loss):
                error_msg = "Training terminated: Final loss is NaN. Loss history: "\
                    f"{history.history['loss']}"
                tf.get_logger().setLevel('INFO')
                raise ValueError(error_msg)

        if self.context.config.input_output.verbose:
            print("Fitted model successfully.")

        tf.get_logger().setLevel('INFO')

        return history

    def decode(
        self,
        data: SequenceDataset,
        indices: np.ndarray,
        batch_size: int,
        models: list[int],
        non_homogeneous_mask_func=None,
    ) -> np.ndarray:
        if tf.distribute.has_strategy():
            with tf.distribute.get_strategy().scope():
                cell_copy = self.msa_hmm_layer.cell.duplicate(models)
        else:
            cell_copy = self.msa_hmm_layer.cell.duplicate(models)

        cell_copy.build(
            (self.context.config.training.num_model, None, None, self.msa_hmm_layer.cell.dim)
        )
        self.encode_only = True
        viterbi_seqs =  viterbi.get_state_seqs_max_lik(
            data,
            self.context.batch_gen,
            indices,
            batch_size,
            cell_copy,
            models,
            self,
            non_homogeneous_mask_func,
            with_plm=self.context.config.language_model.use_language_model,
            plm_dim=self.context.scoring_model_config.dim
        )
        self.encode_only = False
        return viterbi_seqs

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

    def loglik(self, y_pred):
        """Extract log-likelihood from predictions."""
        return y_pred[1]

    def prior(self, y_pred):
        """Extract prior from predictions."""
        return tf.reduce_mean(y_pred[2])

    def aux_loss(self, y_pred):
        """Extract auxiliary loss from predictions."""
        return y_pred[3]

    def compute_loss(self, x, y, y_pred, sample_weight):
        """
        Compute the total loss including likelihood, prior, and auxiliary losses.
        """
        if len(y_pred) == 4:
            loss = -self.loglik(y_pred) - self.prior(y_pred) + self.aux_loss(y_pred)
        else:
            loss = -self.loglik(y_pred)
        loss += sum(self.losses)
        self.loss_tracker.update_state(loss)
        return loss

    def compute_metrics(self, x, y, y_pred, sample_weight):
        """
        Compute and return metrics for monitoring training.
        """
        metric_results = {"loss": self.loss_tracker.result()}
        self.loglik_tracker.update_state(self.loglik(y_pred))
        metric_results["loglik"] = self.loglik_tracker.result()
        if len(y_pred) == 4:
            self.use_prior = True
            self.prior_tracker.update_state(self.prior(y_pred))
            self.aux_loss_tracker.update_state(self.aux_loss(y_pred))
            metric_results["prior"] = self.prior_tracker.result()
            metric_results["aux_loss"] = self.aux_loss_tracker.result()
        return metric_results

    def reset_metrics(self):
        """Reset all metric trackers."""
        self.loss_tracker.reset_state()
        self.loglik_tracker.reset_state()
        self.prior_tracker.reset_state()
        self.aux_loss_tracker.reset_state()

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


class PermuteSeqs(tf.keras.layers.Layer):
    """Layer for transposing tensor dimensions."""

    def __init__(self, perm, **kwargs):
        super(PermuteSeqs, self).__init__(**kwargs)
        self.perm = perm

    def call(self, sequences):
        return tf.transpose(sequences, self.perm)

    def get_config(self):
        config = super().get_config()
        config.update({"perm": self.perm})
        return config


class Identity(tf.keras.layers.Layer):
    """Identity layer that passes input through unchanged."""

    def call(self, x):
        return x

    def get_config(self):
        return super().get_config()


# Register custom objects for serialization
tf.keras.utils.get_custom_objects()["LearnMSAModel"] = LearnMSAModel
tf.keras.utils.get_custom_objects()["PermuteSeqs"] = PermuteSeqs
tf.keras.utils.get_custom_objects()["Identity"] = Identity