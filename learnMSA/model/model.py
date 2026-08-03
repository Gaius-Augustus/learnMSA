"""Backend-neutral learnMSA model.

:class:`LearnMSAModel` holds the parts of the model that are the same whichever
tensor framework runs it: the batching and epoch heuristics, serialization to
and from a configuration, the derived likelihood/AIC estimates, and the console
reporting.

Everything that touches tensors -- the forward pass, the training loop, the
prediction pipeline -- is abstract here and implemented by the backend, which
subclasses this alongside its framework model type, mirroring hidten::

    class TFLearnMSAModel(tf.keras.Model, LearnMSAModel[T_TFTensor]):
        ...
"""

import math
import sys
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Generic, Literal, Sequence, TypeVar

import numpy as np

import learnMSA.backend as backend
from learnMSA.model.context import LearnMSAContext
from learnMSA.model.phmm_mixin import PHMMMixin
from learnMSA.util.clustering import write_sequence_weights
from learnMSA.util.sequence_dataset import Dataset, SequenceDataset

T_Tensor = TypeVar("T_Tensor")


class LearnMSAModel(PHMMMixin, Generic[T_Tensor]):
    """
    The main model class for LearnMSA, combining a pHMM layer with
    ancestoral probability encoding.
    Provides methods for training, evaluation, and prediction.
    """

    context: LearnMSAContext
    """Data-dependent configuration this model was built from."""

    anc_probs_layer: Any = None
    """The ancestral probabilities layer, if evolutionary times are modeled."""

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
            loglik = self.predict(data, indices=indices, models=models)
            assert isinstance(loglik, np.ndarray)
            return loglik
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
            num_gpu = backend.num_gpus()
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
        history: Any
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

    # --- backend-specific -------------------------------------------------
    # Implemented by the backend subclass; listed here as the contract that a
    # backend has to satisfy.

    @abstractmethod
    def build(self, input_shapes=((None,),)) -> None: ...

    @abstractmethod
    def compile(self, total_steps: int | None = None) -> None: ...

    @abstractmethod
    def fit(self, data, indices=None, iteration=0, batch_size=None,
            epochs=None, steps_per_epoch=None, callbacks=None): ...

    @abstractmethod
    def predict(self, data, indices=None, **kwargs): ...

    @abstractmethod
    def evaluate(self, data, indices=None, models=None): ...

    @abstractmethod
    def compute_null_model_log_probs(self, data, background_dist=None,
                                     transition_prob=None): ...

    @abstractmethod
    def compute_consensus_score(self) -> np.ndarray: ...

    @abstractmethod
    def get_train_callbacks(self, iteration: int = 0) -> list: ...

    @abstractmethod
    def save(self, filepath) -> None:
        """Persist the model; the format is backend specific."""


def make_learnmsa_model(context: LearnMSAContext, **kwargs) -> LearnMSAModel:
    """Construct a learnMSA model for the selected backend."""
    from learnMSA.backend import resolve
    return resolve("model.model", "LearnMSAModel")(context, **kwargs)
