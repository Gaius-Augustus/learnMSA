import time
from pathlib import Path
from typing import Sequence

import numpy as np

from learnMSA.align.align_hits import HitAlignmentMode
from learnMSA.align.alignment_metadata import AlignmentMetaData
from learnMSA.align.alignment_model import AlignmentModel
from learnMSA.model.batch_generator import (get_index_table,
                                            sort_index_table)
from learnMSA.model.model import LearnMSAModel
from learnMSA.util.dataset import Dataset
from learnMSA.util.multi_dataset import MultiSequenceDataset


class MultiAlignmentModel(AlignmentModel):
    """
    Alignments of several datasets that were trained jointly, one model
    (head) per dataset (see :func:`learnMSA.align.align.align_batch`).

    Head k aligns the global indices ``head_indices[k]`` of a
    :class:`~learnMSA.util.multi_dataset.MultiSequenceDataset`. All heads are
    decoded in the same pass over the model; only the post-processing of the
    decoded state sequences runs per head.

    The output methods (``to_file``, ``to_string``, ...) take the head index
    as ``model_index`` and write the alignment of that head's dataset. They
    build the alignments of all heads first if needed. :meth:`head` returns
    a plain AlignmentModel for a single dataset.
    """

    head_indices: list[np.ndarray]
    """The global sequence indices aligned by each head."""

    def __init__(
        self,
        data: MultiSequenceDataset
            | tuple[MultiSequenceDataset, *tuple[Dataset, ...]],
        model: LearnMSAModel,
        head_indices: Sequence[np.ndarray],
        gap_symbol: str = '-',
        gap_symbol_insertions: str = '.',
        hit_alignment_mode: HitAlignmentMode = HitAlignmentMode.GREEDY_SCORES,
    ) -> None:
        """
        Args:
            data: The MultiSequenceDataset, optionally followed by auxiliary
                MultiDatasets.
            model: A learnMSA model with one head per dataset.
            head_indices: For every head, the global indices of the
                sequences it aligns.
            gap_symbol: Character used to denote missing match positions.
            gap_symbol_insertions: Character used to denote insertions in
                other sequences.
            hit_alignment_mode: Mode for aligning the domain hits.
        """
        self.head_indices = [
            np.asarray(i, dtype=np.int64) for i in head_indices
        ]
        if len(self.head_indices) != model.heads:
            raise ValueError(
                f"Expected indices for {model.heads} heads, got "
                f"{len(self.head_indices)}."
            )
        if any(i.size == 0 for i in self.head_indices):
            raise ValueError("Every head needs at least one sequence.")
        super().__init__(
            data,
            model,
            np.concatenate(self.head_indices),
            gap_symbol=gap_symbol,
            gap_symbol_insertions=gap_symbol_insertions,
            hit_alignment_mode=hit_alignment_mode,
        )

    def head(self, k: int) -> AlignmentModel:
        """An AlignmentModel of head k and its dataset that shares the data,
        the model and (if built) the alignment of this object."""
        am = AlignmentModel(
            self.data,
            self.model,
            self.head_indices[k],
            gap_symbol=self.gap_symbol,
            gap_symbol_insertions=self.gap_symbol_insertions,
            best_head=k,
            hit_alignment_mode=self.hit_alignment_mode,
        )
        if k in self.metadata:
            am.metadata[k] = self.metadata[k]
        return am

    def select_best(self) -> None:
        raise ValueError(
            "Every head of a MultiAlignmentModel aligns its own dataset, "
            "there is no best head to select."
        )

    def build_alignment(
        self,
        models: Sequence[int] | None = None,
        decoding_mode: AlignmentModel.DecodingMode
            = AlignmentModel.DecodingMode.VITERBI,
    ) -> None:
        """Decodes the alignments of the given heads (default: all) in
        parallel.

        Args:
            models: The heads to decode.
            decoding_mode: The mode used for decoding the alignments.
        """
        if models is None:
            models = list(range(len(self.head_indices)))
        models = list(models)

        if self.model.context.config.input_output.verbose:
            print(
                f"Building alignments for {len(models)} models with "
                f"decoding mode {decoding_mode}..."
            )

        self._set_decoding_mode(decoding_mode)

        indices = [self.head_indices[k] for k in models]
        outputs, positions = self._predict_heads(indices, models)

        t = time.time()

        metas = [
            self._stack_head_outputs(outputs, c, k, positions)
            for c, k in enumerate(models)
        ]
        del outputs

        rows = [self._multi_hit_rows(meta) for meta in metas]
        singles: list[AlignmentMetaData | None] = [None] * len(models)
        rerun = [c for c, r in enumerate(rows) if r is not None]
        if rerun:
            # Re-run Viterbi for all sequences with more than one hit
            # and a model where multi-hits are forbidden, in one pass
            self.model.phmm_layer.enable_multi_hits(False)
            rerun_models = [models[c] for c in rerun]
            outputs, positions = self._predict_heads(
                [indices[c][rows[c]] for c in rerun], rerun_models
            )
            for j, c in enumerate(rerun):
                singles[c] = self._stack_head_outputs(
                    outputs, j, rerun_models[j], positions
                )
            del outputs
            # Restore original behavior
            self.model.phmm_layer.enable_multi_hits(True)

        for c, k in enumerate(models):
            self.metadata[k] = self._align_hits(metas[c], rows[c], singles[c])

        if self.model.context.config.input_output.verbose:
            print(
                f"Building alignments took {time.time() - t:.2f} seconds."
            )

    def _predict_heads(
        self,
        indices: Sequence[np.ndarray],
        models: Sequence[int],
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Runs every model on its own sequences in one prediction pass.

        Returns:
            The ragged outputs of ``predict`` and the positions table of
            :func:`sort_index_table`.
        """
        table, positions = sort_index_table(
            get_index_table(indices), self.data[0].seq_lens
        )
        outputs = self.model.predict(
            self.data, indices=table, models=list(models), ragged_output=True
        )
        return outputs, positions

    def _stack_head_outputs(
        self,
        outputs: list[tuple[np.ndarray, np.ndarray]],
        column: int,
        model: int,
        positions: np.ndarray,
    ) -> AlignmentMetaData:
        """Decodes the state sequences of one model column of a joint pass.
        """
        model_len = self.model.context.model_lengths[model]
        all_meta_data = []
        all_idx = []
        for seqs, rows in outputs:
            pos = positions[rows, column]
            keep = pos >= 0
            if not np.any(keep):
                continue
            all_meta_data.append(
                AlignmentModel.decode(model_len, seqs[keep, :, column])
            )
            all_idx.append(pos[keep])
        meta_data = AlignmentMetaData.concat(all_meta_data)
        meta_data.reorder(np.concatenate(all_idx))
        return meta_data

    def _head_view(
        self,
        model_index: int,
        decoding_mode: AlignmentModel.DecodingMode,
    ) -> AlignmentModel:
        if model_index not in self.metadata:
            # Decode all heads at once rather than one after the other
            self.build_alignment(None, decoding_mode)
        return self.head(model_index)

    def to_string(self, model_index: int, *args, **kwargs) -> list[str]:
        mode = kwargs.get("decoding_mode", AlignmentModel.DecodingMode.VITERBI)
        return self._head_view(model_index, mode).to_string(
            model_index, *args, **kwargs
        )

    def to_file(
        self, filepath: str | Path, model_index: int, *args, **kwargs
    ) -> Path:
        mode = kwargs.get("decoding_mode", AlignmentModel.DecodingMode.VITERBI)
        return self._head_view(model_index, mode).to_file(
            filepath, model_index, *args, **kwargs
        )

    def estimate_fasta_size(self, model_index: int, *args, **kwargs) -> int:
        mode = kwargs.get("decoding_mode", AlignmentModel.DecodingMode.VITERBI)
        return self._head_view(model_index, mode).estimate_fasta_size(
            model_index, *args, **kwargs
        )

    def get_batch_alignment(
        self, model_index: int, *args, **kwargs
    ) -> np.ndarray:
        mode = kwargs.get("decoding_mode", AlignmentModel.DecodingMode.VITERBI)
        return self._head_view(model_index, mode).get_batch_alignment(
            model_index, *args, **kwargs
        )

    def get_batch_states(
        self, model_index: int, *args, **kwargs
    ) -> np.ndarray:
        mode = kwargs.get("decoding_mode", AlignmentModel.DecodingMode.VITERBI)
        return self._head_view(model_index, mode).get_batch_states(
            model_index, *args, **kwargs
        )

    def states_to_string(self, model_index: int, *args, **kwargs) -> list[str]:
        mode = kwargs.get("decoding_mode", AlignmentModel.DecodingMode.VITERBI)
        return self._head_view(model_index, mode).states_to_string(
            model_index, *args, **kwargs
        )

    def states_to_file(
        self, filepath: str | Path, model_index: int, *args, **kwargs
    ) -> None:
        mode = kwargs.get("decoding_mode", AlignmentModel.DecodingMode.VITERBI)
        self._head_view(model_index, mode).states_to_file(
            filepath, model_index, *args, **kwargs
        )

    def write_scores(self, filepath: Path, model: int) -> None:
        raise NotImplementedError(
            "write_scores is not supported for a MultiAlignmentModel."
        )
