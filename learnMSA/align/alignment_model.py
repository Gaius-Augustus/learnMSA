import json
import shutil
import time
import warnings
from enum import Enum
from pathlib import Path

import numpy as np
import tensorflow as tf

from learnMSA.align.align_inserts import AlignedInsertions
from learnMSA.align.alignment_metadata import AlignmentMetaData
from learnMSA.align.align_hits import HitAlignmentMode, hit_alignment
from learnMSA.model.select import SelectionCriterion, select_model
from learnMSA.model.tf.model import LearnMSAModel
from learnMSA.util.aligned_dataset import AlignedDataset, SequenceDataset
from learnMSA.util.dataset import Dataset


class AlignmentModel():
    """
    Decodes alignments from a number of models, stores them in a memory
    friendly representation and generates table-form (memory unfriendly)
    alignments on demand (batch-wise mode possible).
    """

    data: tuple[SequenceDataset, *tuple[Dataset, ...]]
    """The dataset of sequences."""

    model: LearnMSAModel
    """A learnMSA model instance for decoding and scoring."""

    indices: np.ndarray
    """Array of sequence indices specifying which sequences from data are
    included."""

    gap_symbol: str
    """Character used to denote missing match positions."""

    gap_symbol_insertions: str
    """Character used to denote insertions in other sequences."""

    metadata: dict[int, AlignmentMetaData]
    """Alignment metadata for each model."""

    best_head: int
    """Index of the head that best fits the training data."""

    hit_alignment_mode: HitAlignmentMode
    """Mode for aligning the domain hits."""
    class DecodingMode(Enum):
        VITERBI = "viterbi"
        MEA = "mea"

        @staticmethod
        def from_str(label: str) -> 'AlignmentModel.DecodingMode':
            if label.lower() == 'viterbi':
                return AlignmentModel.DecodingMode.VITERBI
            elif label.lower() == 'mea':
                return AlignmentModel.DecodingMode.MEA
            else:
                raise ValueError(f"Unsupported decoding mode: {label}")


    def __init__(
        self,
        data: SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
        model: LearnMSAModel,
        indices: np.ndarray|None = None,
        gap_symbol: str = '-',
        gap_symbol_insertions: str = '.',
        best_head: int = -1,
        hit_alignment_mode: HitAlignmentMode = HitAlignmentMode.GREEDY_CONSENSUS,
    ) -> None:
        """
        Args:
            data: The dataset of sequences. Can be a SequenceDataset or a tuple
                of datasets where the first entry is a SequenceDataset.
            model: A learnMSA model instance for decoding and scoring.
            indices: An optional array of sequence indices specifying which
                sequences from data are included in the alignment. If None,
                all sequences are included.
            gap_symbol: Character used to denote missing match positions.
            gap_symbol_insertions: Character used to denote insertions in other
                sequences.
            best_head: Index of the head that best fits the training data.
                Defaults to -1 (no model selected).
            hit_alignment_mode: Mode for aligning the domain hits.
        """
        if isinstance(data, SequenceDataset):
            data = (data,)
        self.data = data
        self.model = model
        if indices is None:
            self.indices = np.arange(data[0].num_seq)
        else:
            self.indices = indices
        self.gap_symbol = gap_symbol
        self.gap_symbol_insertions = gap_symbol_insertions
        self.best_head = best_head
        self.hit_alignment_mode = hit_alignment_mode
        self.metadata = {}

    def get_output_alphabet(self, a2m: bool = True) -> np.ndarray:
        """ Returns the output alphabet used for string representation of
            alignments.

        Args:
            a2m (bool): Whether to use the a2m format for strings
                (with lowercase letters for inserted amino acids and dots for
                gaps in insertions).
        """
        if a2m:
            output_alphabet = np.array((
                list(self.data[0].get_alphabet_no_gap()) +
                [self.gap_symbol] +
                list(self.data[0].get_alphabet_no_gap().lower()) +
                [self.gap_symbol_insertions, "$"]
            ))
        else:
            output_alphabet = np.array((
                list(self.data[0].get_alphabet_no_gap()) +
                [self.gap_symbol] +
                list(self.data[0].get_alphabet_no_gap()) +
                [self.gap_symbol, "$"]
            ))
        return output_alphabet

    def select_best(self) -> None:
        criterion = SelectionCriterion(
            self.model.context.config.training.model_criterion
        )
        self.best_head = select_model(
            self.model,
            self.data,
            criterion,
            sequence_indices=self.indices,
            verbose=self.model.context.config.input_output.verbose,
        )

    def to_string(
        self,
        model_index: int,
        add_block_sep: bool = True,
        aligned_insertions: AlignedInsertions = AlignedInsertions(),
        a2m: bool = True,
        only_matches: bool = False,
        decoding_mode: DecodingMode = DecodingMode.VITERBI,
    ) -> list[str]:
        """ Select one model and decode an alignment that is returned as a
            list of strings.
            Note that this method is not suitable if memory is limited and
            alignment depths and width are large.

        Args:
            model_index: Specifies the model for decoding. Use a suitable
                criterion like loglik to decide for a model.
            add_block_sep: If true, columns containing a special character are
                added to the alignment indicating domain boundaries.
            aligned_insertions: Can be used to override insertion metadata if
                insertions are aligned after the main procedure.
            a2m: Whether to use the a2m format for strings
                (with lowercase letters for inserted amino acids and dots for
                gaps in insertions).
            only_matches: If true, omit all insertions and write only those
                amino acids that are assigned to match states.
            decoding_mode: The mode used for decoding the alignment.
        """
        output_alphabet = self.get_output_alphabet(a2m)
        batch_alignment = self.get_batch_alignment(
            model_index=model_index,
            batch_indices=np.arange(self.indices.size),
            add_block_sep=add_block_sep,
            aligned_insertions=aligned_insertions,
            only_matches=only_matches,
            decoding_mode=decoding_mode,
        )
        alignment_strings = self.batch_to_string(
            batch_alignment, output_alphabet=output_alphabet
        )
        return alignment_strings

    def to_file(
        self,
        filepath: str | Path,
        model_index: int,
        batch_size: int = 100000,
        add_block_sep: bool = False,
        aligned_insertions : AlignedInsertions = AlignedInsertions(),
        format: str = "fasta",
        fasta_line_limit: int = 80,
        only_matches: bool = False,
        decoding_mode: DecodingMode = DecodingMode.VITERBI,
    ) -> None:
        """ Select one model and decode an alignment that is written in fasta
            or a2m file format.
            The file is written batch wise. The memory required for this
            operation must be large enough to hold decode and store a single
            batch of aligned sequences but not the whole alignment.
        Args:
            model_index: Specifies the model for decoding. Use a suitable
                criterion like loglik to decide for a model.
            batch_size: Defines how many sequences are decoded into table form
                and written to file at a time.
            add_block_sep: If true, columns containing a special character are
                added to the alignment indicating domain boundaries.
            aligned_insertions: Can be used to override insertion metadata if
                insertions are aligned after the main procedure.
            format: Output format. Important for large data: learnMSA is only
                able to stream fasta files.
                Other formats require a conversion, i.e. the whole alignment is
                stored in memory.
            fasta_line_limit: Maximum number of characters per line in the
                fasta file (only applies to sequences).
            only_matches: If true, omit all insertions and write only those
                amino acids that are assigned to match states.
            decoding_mode: The mode used for decoding the alignment.
        """
        if format == "fasta" or format == "a2m":
            # Stream batches to file
            output_alphabet = self.get_output_alphabet(format == "a2m")
            # Use a large write buffer only when the output is large enough to
            # benefit from it. Total residues is a lower bound on output size.
            total_residues = int(np.sum(self.data[0].seq_lens[self.indices]))
            write_buffer = 8 * 1024 * 1024 if total_residues > 1_000_000 else -1
            with open(filepath, "w", buffering=write_buffer) as output_file:
                n = self.indices.size
                i = 0
                while i < n:
                    batch_indices = np.arange(i, min(n, i+batch_size))
                    batch_alignment = self.get_batch_alignment(
                        model_index=model_index,
                        batch_indices=batch_indices,
                        add_block_sep=add_block_sep,
                        aligned_insertions=aligned_insertions,
                        only_matches=only_matches,
                        decoding_mode=decoding_mode,
                    )
                    alignment_strings = self.batch_to_string(
                        batch_alignment, output_alphabet=output_alphabet
                    )
                    entries = []
                    for s, seq_ind in zip(alignment_strings, batch_indices):
                        seq_header = self.data[0].get_header(
                            self.indices[seq_ind]
                        )
                        entry = ">"+seq_header+"\n"
                        entry += "\n".join(
                            s[j:j+fasta_line_limit]
                            for j in range(0, len(s), fasta_line_limit)
                        )
                        entry += "\n"
                        entries.append(entry)
                    output_file.writelines(entries)
                    i += batch_size
        else:
            # Decode the whole alignment into memory and write the entire
            # thing at once
            msa = self.to_string(
                model_index, add_block_sep, aligned_insertions
            )
            msa = [
                (self.data[0].seq_ids[self.indices[i]], msa[i])
                for i in range(len(msa))
            ]
            data = AlignedDataset(sequences=msa)
            data.write(filepath, format)

    def get_batch_alignment(
        self,
        model_index: int,
        batch_indices: np.ndarray,
        add_block_sep: bool = True,
        aligned_insertions: AlignedInsertions = AlignedInsertions(),
        only_matches: bool = False,
        decoding_mode: DecodingMode = DecodingMode.VITERBI,
    ) -> np.ndarray:
        """ Returns a dense matrix representing a subset of sequences
            as specified by batch_indices with respect to the alignment of all
            sequences (i.e. the sub alignment can contain gap-only columns and
            stacking all batches yields a complete alignment).
        Args:
            model_index: Specifies the model for decoding. Use a suitable
                criterion like loglik to decide for a model.
            batch_indices: Sequence indices / indices of alignment rows.
            add_block_sep: If true, columns containing a special character are
                added to the alignment indicating domain boundaries.
            aligned_insertions: Can be used to override insertion metadata if
                insertions are aligned after the main procedure.
            only_matches: If true, omit all insertions and write only those
                amino acids that are assigned to match states.
            decoding_mode: The mode used for decoding the alignment.
        """
        if not model_index in self.metadata:
            self.build_alignment([model_index], decoding_mode)
        meta_data = self.metadata[model_index]

        # Gather the sequences for the batch
        b = batch_indices.size
        sequences = np.zeros((b, self.data[0].max_len), dtype=np.uint16)
        sequences += (len(self.data[0].alphabet)-1)
        for i,j in enumerate(batch_indices):
            idx = int(self.indices[j])
            l = self.data[0].seq_lens[idx]
            sequences[i, :l] = self.data[0].get_encoded_seq(idx)

        # Construct the alignment blocks
        blocks = []
        if add_block_sep:
            sep = np.zeros((b,1), dtype=np.uint16) + 2*len(self.data[0].alphabet)

        # Left flank
        if not only_matches:
            left_flank_block = self.get_insertion_block(
                sequences,
                meta_data.left_flank_len_for(batch_indices),
                max(
                    meta_data.left_flank_len_total,
                    aligned_insertions.ext_left_flank
                ),
                meta_data.left_flank_start_for(batch_indices),
                adjust_to_right=True,
                custom_columns=aligned_insertions.left_flank(batch_indices)
            )
            blocks.append(left_flank_block)
            if add_block_sep:
                blocks.append(sep)

        # Pre-compute which match-state columns are non-empty across all rows.
        is_non_empty_all = meta_data.repeat_occupancy_mask()  # (num_repeats, num_match)

        for i in range(meta_data.num_repeats):
            dh_batch, il_batch, is_batch, _, _ = meta_data.get_repeat_data(
                i, batch_indices
            )

            # One domain hit
            alignment_block = self.get_alignment_block(
                sequences=sequences,
                consensus=dh_batch,
                ins_len=il_batch,
                ins_len_total=np.maximum(
                    meta_data.insertion_lens_total,
                    aligned_insertions.ext_insertions
                )[i],
                ins_start=is_batch,
                is_non_empty=is_non_empty_all[i],
                custom_columns=aligned_insertions.insertion(batch_indices, i),
                only_matches=only_matches
            )
            blocks.append(alignment_block)

            if add_block_sep:
                blocks.append(sep)

            # Unannotated segment (if there are more repeats to come)
            if i < meta_data.num_repeats - 1 and not only_matches:
                uns_l, uns_s = meta_data.get_unannotated_data(i, batch_indices)
                unannotated_block = self.get_insertion_block(
                    sequences,
                    uns_l,
                    np.maximum(
                        meta_data.unannotated_segment_lens_total,
                        aligned_insertions.ext_unannotated
                    )[i],
                    uns_s,
                    custom_columns=aligned_insertions.unannotated_segment(
                        batch_indices, i
                    )
                )
                blocks.append(unannotated_block)
                if add_block_sep:
                    blocks.append(sep)

        # Right flank
        if not only_matches:
            right_flank_block = self.get_insertion_block(
                sequences,
                meta_data.right_flank_len_for(batch_indices),
                max(
                    meta_data.right_flank_len_total,
                    aligned_insertions.ext_right_flank
                ),
                meta_data.right_flank_start_for(batch_indices),
                custom_columns=aligned_insertions.right_flank(batch_indices)
            )
            blocks.append(right_flank_block)

        batch_alignment = np.concatenate(blocks, axis=1)
        return batch_alignment

    def batch_to_string(
        self, batch_alignment: np.ndarray, output_alphabet: np.ndarray
    ) -> list[str]:
        """ Converts a dense matrix into string format.
        """
        chars = output_alphabet[batch_alignment]  # (n, L) dtype '<U1'
        # View each row as a single string (requires C-contiguous, fixed-width chars)
        n, L = chars.shape
        return np.ascontiguousarray(chars).view(f'U{L}').reshape(n).tolist()

    def write_scores(self, filepath: Path, model: int) -> None:
        """ Writes per-sequence scores (loglik, bitscore) to a
            tsv file sorted by the bitscore ``loglik(S) - log P(S; nullmodel)``.
        Args:
            filepath: Path of the output file.
            model: The model for which scores are written.
        """

        # Disable ancestral probabilities as they are specific for the
        # training sequences and will not apply to target sequences
        _anc_probs = self.model.context.config.training.use_anc_probs
        self.model.context.config.training.use_anc_probs = False

        # Compute the likelihood and bitscores for all sequences
        loglik = self.model.estimate_loglik(
            self.data, self.data[0].num_seq, reduce=False, models=[model]
        )[:,0]
        # Compute the bitscore
        A = self.model.phmm_layer.hmm.transitioner.matrix()
        B = self.model.phmm_layer.hmm.emitter[0].matrix()
        L = self.model.lengths[model]
        log_null = self.model.compute_null_model_log_probs(
            self.data[0],
            background_dist=B[model, L],
            transition_prob=A[model, 2*L-1, 2*L-1]
        )
        bitscore = (loglik - log_null) / np.log(2.0)

        # Sort by bitscore in descending order
        sorted_indices = np.argsort(-bitscore)

        # Write to file
        with open(filepath, "w") as scorefile:
            scorefile.write(
                "\t".join(["seq_id", "loglik", "bit_score"]) + "\n"
            )
            for idx in sorted_indices:
                scorefile.write("\t".join([
                    f"{self.data[0].seq_ids[idx]}",
                    f"{loglik[idx]}",
                    f"{bitscore[idx]}"
                ]) + "\n")

        # Restore the original setting for ancestral probabilities
        self.model.context.config.training.use_anc_probs = _anc_probs

    def save(self, filepath: str | Path, pack: bool = True) -> None:
        """ Writes the underlying models to file.

        Args:
            filepath: Path of the written file.
            pack: If true, the output will be a zip file, otherwise a directory.
        """
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)
        # Serialize metadata
        d: dict = {
            "gap_symbol" : self.gap_symbol,
            "gap_symbol_insertions" : self.gap_symbol_insertions,
            "best_head" : getattr(self, "best_head", None),
        }
        with open(filepath / "meta.json", "w") as metafile:
            metafile.write(json.dumps(d, indent=4))
        # Serialize indices
        np.savetxt(filepath / "indices", self.indices, fmt='%i')
        # Save the model
        self.model.save(str(filepath) + ".keras")
        if pack:
            shutil.make_archive(str(filepath), "zip", filepath)
            try:
                shutil.rmtree(filepath)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        data: SequenceDataset | tuple[SequenceDataset, *tuple[Dataset, ...]],
        from_packed: bool = True,
    ):
        """ Loads an AlignmentModel instance from a file.

        Args:
            filepath: Path of the file to load.
            from_packed: Pass true or false depending on the pack argument used
                with write_models_to_file.

        Returns:
            An AlignmentModel instance with equivalent behavior as the
            AlignmentModel instance used while saving the model.
        """
        filepath = Path(filepath)
        if from_packed:
            shutil.unpack_archive(str(filepath) + ".zip", filepath)

        # Deserialize metadata
        with open(filepath / "meta.json") as metafile:
            d = json.load(metafile)

        # Deserialize indices
        indices = np.loadtxt(filepath / "indices", dtype=int)

        # Load the model
        with warnings.catch_warnings():
            # Suppress the compile warning since we manually compile right after
            warnings.filterwarnings(
                'ignore',
                message=".*compile.*was not called as part of model loading.*",
                category=UserWarning
            )
            model = tf.keras.models.load_model(
                str(filepath) + ".keras",
            )

        # Manually compile the model after loading
        model.compile()

        if from_packed:
            #after loading remove unpacked files and keep only the archive
            try:
                shutil.rmtree(filepath)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        am = cls(
            data,
            model,
            indices,
            d["gap_symbol"],
            d["gap_symbol_insertions"],
            d.get("best_head", -1),
        )
        return am


    @classmethod
    def decode_core(cls, model_length, state_seqs_max_lik, indices):
        """
        Decodes consensus columns as a matrix as well as insertion lengths
        and starting positions as auxiliary vectors.

        Args:
            model_length: Number of match states
                (length of the consensus sequence).
            state_seqs_max_lik: A tensor with the most likeli state sequences.
                Shape: (num_seq, L)
            indices: Indices in the sequences where decoding should start.
                Shape: (num_seq)
        Returns:
            consensus_columns: Decoded consensus columns.
                Shape: (num_seq, model_length)
            insertion_lens: Number of amino acids emitted per insertion state.
                Shape: (num_seq, model_length-1)
            insertion_start: Starting position of each insertion in the
                sequences. Shape: (num_seq, model_length-1)
            finished: Boolean vector indicating sequences that are fully
                decoded. Shape: (num_seq)
        """
        n, T = state_seqs_max_lik.shape
        c = model_length
        consensus_columns = -np.ones((n, c), dtype=np.int16)
        insertion_lens    = np.zeros((n, c-1), dtype=np.int16)
        insertion_start   = -np.ones((n, c-1), dtype=np.int16)

        pos    = np.arange(T)
        active = pos[None, :] >= indices[:, None]  # (n, T): positions in scope
        s      = state_seqs_max_lik

        # Locate the terminal state (unannotated or end) for each sequence
        is_unannotated = active & (s == 2*c)
        is_at_end      = active & ((s == 2*c+1) | (s == 2*c+2))
        is_terminal    = is_unannotated | is_at_end
        has_terminal   = np.any(is_terminal, axis=1)                           # (n,)
        end_pos        = np.where(has_terminal, np.argmax(is_terminal, axis=1), T)  # (n,)

        # Only process positions inside the core region [indices[i], end_pos[i])
        in_core   = active & (pos[None, :] < end_pos[:, None])
        is_match  = in_core & (s >= 0)   & (s < c)
        is_insert = in_core & (s >= c)   & (s < 2*c - 1)

        # Consensus columns: record which sequence position fills each match state
        seq_idx_m, pos_idx_m = np.where(is_match)
        consensus_columns[seq_idx_m, s[seq_idx_m, pos_idx_m]] = pos_idx_m.astype(np.int16)

        # Insertion lengths (count per seq × insert-state)
        seq_idx_i, pos_idx_i = np.where(is_insert)
        insert_states = s[seq_idx_i, pos_idx_i] - c
        np.add.at(insertion_lens, (seq_idx_i, insert_states), 1)

        # Insertion starts: minimum position per (seq, insert-state)
        ins_start_tmp = np.full((n, c-1), T, dtype=np.int32)
        np.minimum.at(ins_start_tmp, (seq_idx_i, insert_states), pos_idx_i)
        has_insert = ins_start_tmp < T
        insertion_start[has_insert] = ins_start_tmp[has_insert].astype(np.int16)

        # finished[i] = True when terminal is an end state, not unannotated
        ep_clamped = np.minimum(end_pos, T - 1)
        finished   = has_terminal & is_at_end[np.arange(n), ep_clamped]

        indices[:] = end_pos.astype(indices.dtype)
        return consensus_columns, insertion_lens, insertion_start, finished


    @classmethod
    def decode_flank(cls, state_seqs_max_lik, flank_state_id, indices):
        """
        Decodes flanking insertion states. The decoding is active as long
        as at least one sequence remains in a flank/unannotated state.

        Args:
            state_seqs_max_lik: A tensor with the most likeli state sequences.
                Shape: (num_seq, L)
            flank_state_id: Index of the flanking state.
            indices: Indices in the sequences where decoding should start.
                Shape: (num_seq)

        Returns:
            insertion_lens: Number of amino acids emitted per insertion state.
                Shape: (num_seq, model_length-1)
            insertion_start: Starting position of each insertion in the
                sequences. Shape: (num_seq, model_length-1)
        """
        n, T = state_seqs_max_lik.shape
        insertion_start = np.copy(indices)

        # For each sequence find the first position >= indices[i] that is not
        # the flank state; that is where the flank run ends.
        pos       = np.arange(T)
        active    = pos[None, :] >= indices[:, None]  # (n, T)
        non_flank = active & (state_seqs_max_lik != flank_state_id)
        has_non_flank = np.any(non_flank, axis=1)
        end_pos   = np.where(has_non_flank, np.argmax(non_flank, axis=1), T)

        insertion_lens = (end_pos - indices).astype(np.int16)
        indices[:] = end_pos.astype(indices.dtype)
        return insertion_lens, insertion_start


    @classmethod
    def decode(cls, model_length, state_seqs_max_lik) -> AlignmentMetaData:
        """
        Decodes an implicit alignment (insertion start/length are
        represented as 2 integers) from most likely state sequences.

        Args:
            model_length: Number of match states (length of the consensus
                sequence).
            state_seqs_max_lik: A tensor with the most likeli state sequences.
                Shape: (num_seq, L)

        Returns:
            AlignmentMetaData: Object containing the decoded alignment
                information.
        """
        n = state_seqs_max_lik.shape[0]
        c = model_length #alias for code readability
        indices = np.zeros(n, np.int16) # active positions in the sequence

        left_flank = cls.decode_flank(state_seqs_max_lik, 2*c-1, indices)

        core_blocks = []
        insertion_lens = []
        insertion_starts = []
        unannotated_segments = []
        core_starts = []
        core_ends = []
        finished_blocks = []
        while True:
            core_start = np.copy(indices)
            C, IL, IS, finished = cls.decode_core(
                model_length, state_seqs_max_lik, indices
            )
            core_end = np.copy(indices)
            core_blocks.append(C)
            insertion_lens.append(IL)
            insertion_starts.append(IS)
            core_starts.append(core_start)
            core_ends.append(core_end)
            finished_blocks.append(finished)

            if np.all(finished):
                break

            unannotated_segments.append(
                cls.decode_flank(state_seqs_max_lik, 2*c, indices)
            )

        right_flank = cls.decode_flank(state_seqs_max_lik, 2*c+1, indices)

        # Compute num_repeats_per_row from finished_blocks.
        # finished_blocks[r][j] = True means sequence j finishes after repeat r.
        max_R = len(core_blocks)
        finished_stack = np.stack(finished_blocks, axis=0)  # (max_R, n)
        first_finish = np.argmax(finished_stack, axis=0)    # (n,) 0-based index
        num_repeats_per_row = (first_finish + 1).astype(np.int32)

        # Row offsets into flat arrays
        row_offsets = np.concatenate([[0], np.cumsum(num_repeats_per_row)]).astype(np.int32)
        total_R = int(row_offsets[-1])
        M = core_blocks[0].shape[1] if max_R > 0 else 0

        # Allocate flat arrays
        domain_hit_flat  = np.full((total_R, M), -1, dtype=np.int16)
        domain_loc_flat  = np.full((total_R, 2), -1, dtype=np.int16)
        ins_lens_flat    = np.zeros((total_R, max(0, M - 1)), dtype=np.int16)
        ins_start_flat   = np.full((total_R, max(0, M - 1)), -1, dtype=np.int16)

        for r in range(max_R):
            rows = np.where(num_repeats_per_row > r)[0]
            flat_idx = row_offsets[rows] + r
            domain_hit_flat[flat_idx] = core_blocks[r][rows]
            domain_loc_flat[flat_idx, 0] = core_starts[r][rows]
            domain_loc_flat[flat_idx, 1] = core_ends[r][rows]
            ins_lens_flat[flat_idx]  = insertion_lens[r][rows]
            ins_start_flat[flat_idx] = insertion_starts[r][rows]

        # Flat unannotated segments
        # row j with k repeats has k-1 unannotated segments
        num_uns_per_row = np.maximum(num_repeats_per_row - 1, 0)
        uns_offsets = np.concatenate([[0], np.cumsum(num_uns_per_row)]).astype(np.int32)
        total_U = int(uns_offsets[-1])
        uns_len_flat   = np.zeros(total_U, dtype=np.int16)
        uns_start_flat = np.full(total_U, -1, dtype=np.int16)
        for r, (seg_len, seg_start) in enumerate(unannotated_segments):
            rows = np.where(num_repeats_per_row > r + 1)[0]
            flat_idx = uns_offsets[rows] + r
            uns_len_flat[flat_idx]   = seg_len[rows]
            uns_start_flat[flat_idx] = seg_start[rows]

        return AlignmentMetaData(
            num_rows           = n,
            num_match          = c,
            num_repeats_per_row= num_repeats_per_row,
            domain_hit         = domain_hit_flat,
            domain_loc         = domain_loc_flat,
            insertion_lens     = ins_lens_flat,
            insertion_start    = ins_start_flat,
            left_flank_len     = left_flank[0],
            left_flank_start   = left_flank[1],
            right_flank_len    = right_flank[0],
            right_flank_start  = right_flank[1],
            unannotated_segments_len   = uns_len_flat,
            unannotated_segments_start = uns_start_flat,
        )



    @classmethod
    def get_insertion_block(
        cls,
        sequences,
        lens,
        maxlen,
        starts,
        adjust_to_right=False,
        custom_columns=None
    ):
        """ Constructs one insertion block from an implicitly represented
        alignment.

        Args:
        Returns:
        """
        n = sequences.shape[0]
        A = np.arange(n)
        s = len(SequenceDataset._default_alphabet)
        block = np.zeros((n, maxlen), dtype=np.uint8) + s - 1
        count_down_lens = np.copy(lens)
        active = count_down_lens > 0
        i = 0
        columns = np.stack([np.arange(maxlen)]*n)
        if custom_columns is not None:
            columns[:, :custom_columns.shape[1]] = custom_columns
        while np.any(active):
            aa = sequences[A[active], starts[active] + i]
            block[active, columns[active,i]] = aa
            count_down_lens -= 1
            active = count_down_lens > 0
            i += 1
        if adjust_to_right and custom_columns is None:
            block_right_aligned = np.zeros_like(block) + s - 1
            for i in range(maxlen):

                block_right_aligned[A, (maxlen-lens+i)%maxlen] = block[:, i]
            block = block_right_aligned
        block += s #lower case
        return block


    @classmethod
    def get_alignment_block(
        cls,
        sequences,
        consensus,
        ins_len,
        ins_len_total,
        ins_start,
        is_non_empty=None,
        custom_columns=None,
        only_matches=False,
    ):
        """
        Constructs one core model hit block from an implicitly represented
        alignment.

        Args:

        Returns:
             block: Shape (num_seq, block_length)
        """
        A = np.arange(sequences.shape[0])
        if only_matches:
            length = consensus.shape[1]
        else:
            length = consensus.shape[1] + np.sum(ins_len_total)
        block = np.zeros((sequences.shape[0], length), dtype=np.uint8)
        block += len(SequenceDataset._default_alphabet) - 1
        i = 0
        columns_to_remove = [] #track empty columns to be removed later
        for c in range(consensus.shape[1]-1):
            column = consensus[:,c]
            ins_l = ins_len[:,c]
            ins_l_total = ins_len_total[c]
            ins_s = ins_start[:,c]
            #one column
            no_gap = column != -1
            block[no_gap,i] = sequences[A[no_gap],column[no_gap]]
            #is this column empty in ALL batches? if yes, mark for removal
            if is_non_empty is not None and not is_non_empty[c]:
                columns_to_remove.append(i)
            i += 1
            #insertion
            if not only_matches:
                if custom_columns is None:
                    custom_column = None
                else:
                    custom_column = custom_columns[c]
                block[:,i:i+ins_l_total] = cls.get_insertion_block(
                    sequences,
                    ins_l,
                    ins_l_total,
                    ins_s,
                    custom_columns=custom_column
                )
                i += ins_l_total
        #final column
        no_gap = consensus[:,-1] != -1
        block[no_gap,i] = sequences[A[no_gap],consensus[:,-1][no_gap]]
        if is_non_empty is not None and not is_non_empty[-1]:
            columns_to_remove.append(i)
        #remove columns that are empty in ALL batches
        block = np.delete(block, columns_to_remove, axis=1)
        return block

    #computes an implicit alignment (without storing gaps)
    #eventually, an alignment with explicit gaps can be written
    #in a memory friendly manner to file
    def build_alignment(self, models, decoding_mode: DecodingMode):

        assert len(models) == 1, "Not implemented for multiple models."

        if decoding_mode == AlignmentModel.DecodingMode.VITERBI:
            self.model.viterbi_decode_mode()
        elif decoding_mode == AlignmentModel.DecodingMode.MEA:
            self.model.mea_decode_mode()
        else:
            raise ValueError(f"Unsupported decoding mode: {decoding_mode}")

        # predict returns a plain dict of arrays when _decode_msa is True.
        # The full state-sequence array is never materialised in CPU memory.
        raw_dict = self.model.predict(
            self.data, indices=self.indices, models=models
        )
        c = int(self.model.phmm_layer.lengths[models[0]])
        meta_data_base = AlignmentMetaData(
            num_rows=len(self.indices), num_match=c, **raw_dict
        )

        # TODO: this is just here to generate empty fixed_viterbi_seqs
        _ = self._clean_up_viterbi_seqs(None, [0])


        t = time.time()

        j = models[0]
        if self.hit_alignment_mode == HitAlignmentMode.GREEDY_CONSENSUS:
            # Use occupancy (number of used match states) as hit score.
            occupancy = meta_data_base.occupancy_matrix()  # (R, N), -1 for empty
            meta_data = hit_alignment(
                meta_data_base, self.hit_alignment_mode, occupancy
            )
        else:
            meta_data = hit_alignment(meta_data_base, self.hit_alignment_mode)
        self.metadata[j] = meta_data

        if self.model.context.config.input_output.verbose:
            print(
                f"Building alignment took {time.time() - t:.2f} "+
                "seconds."
            )

    def _clean_up_viterbi_seqs(self, state_seqs_max_lik, models):

        assert len(models) == 1, "Not implemented for multiple models."

        # TODO
        print("WARNING: Fixing faulty Viterbi sequences is currently not supported. SKIPPING.")
        self.fixed_viterbi_seqs = np.array([], dtype=np.int32)
        return state_seqs_max_lik

        # state_seqs_max_lik has shape (num_model, num_seq, L)
        faulty_sequences = find_faulty_sequences(
            state_seqs_max_lik,
            self.model.context.model_lengths[models[0]],
            self.data.seq_lens[self.indices]
        )
        self.fixed_viterbi_seqs = faulty_sequences
        if faulty_sequences.size > 0:
            # repeat Viterbi with a masking that prevents certain transitions
            # that can cause problems
            fixed_state_seqs = self.model.decode(
                self.data,
                faulty_sequences,
                self.batch_size,
                models,
                non_homogeneous_mask_func,
            )
            if state_seqs_max_lik.shape[-1] < fixed_state_seqs.shape[-1]:
                state_seqs_max_like = np.pad(
                    state_seqs_max_lik,
                    ((0,0),(0,0),(0,fixed_state_seqs.shape[-1]-state_seqs_max_lik.shape[-1])),
                    constant_values=2*self.model.context.model_lengths[0]+2
                )
            state_seqs_max_lik[0,faulty_sequences,:fixed_state_seqs.shape[-1]] = fixed_state_seqs[0]
        return state_seqs_max_lik


@tf.function
def non_homogeneous_mask_func(i, seq_lens, hmm_cell):
    """ Let S = S_1 … S_L be the sequence and M_1 … M_Z the match states.
    In a Viterbi path pi = pi_1 … pi_L prevent transitions such that either
    a) (pi_{i-1}, pi_i) = (M_j, E) and L-i <= Z-j or
    b) (pi_{i-1}, pi_i) = (S, M_j) and i <= j.

    Returns:
        A mask of shape (num_models, batch_size, num_states, num_states)
        indicating allowed transitions.
    """
    k = hmm_cell.num_models
    q = tf.cast(hmm_cell.max_num_states, tf.int32)
    template = tf.ones((1,q,q), dtype=hmm_cell.dtype)
    model_masks = []
    for k,length in enumerate(hmm_cell.length):
        length = tf.cast(length, tf.int32)
        C = 2 * length
        states_left = one_hot_set([C], q, hmm_cell.dtype)
        states_right = one_hot_set([C], q, hmm_cell.dtype)
        allowed_CL_transitions = 1 - one_hot_set(tf.range(i+1, tf.maximum(i+1, length + 1)), q, hmm_cell.dtype)
        number_of_forbidden_match_states = tf.maximum(0, length - seq_lens[k] + i)
        #always allow transitions out of the last match state
        number_of_forbidden_match_states = tf.minimum(length-1, number_of_forbidden_match_states)
        length_mask = 1-tf.cast(tf.sequence_mask(number_of_forbidden_match_states, maxlen=q-1), hmm_cell.dtype)
        allowed_CR_transitions = tf.concat([tf.ones_like(length_mask[:,:1]), length_mask], axis=1)
        mask_left = states_left[...,tf.newaxis] * allowed_CL_transitions[tf.newaxis] # type: ignore
        mask_left += template * (1 - states_left[:,tf.newaxis])
        mask_right = states_right[tf.newaxis] * allowed_CR_transitions[...,tf.newaxis] # type: ignore
        mask_right += template * (1 - states_right[tf.newaxis])
        mask = mask_left * mask_right
        model_masks.append(mask)
    return tf.stack(model_masks, axis=0)


@tf.function
def one_hot_set(indices, d, dtype):
    # Returns a vector in {0,1}}^d with a 1 at positions i in indices and 0
    # elsewhere
    return tf.reduce_sum(tf.one_hot(indices, d, dtype=dtype), axis=0)


def find_faulty_sequences(
    state_seqs_max_lik, model_length, seq_lens, limit=32000
):
    if state_seqs_max_lik.shape[1] > limit:
        return np.array([], dtype=np.int32)
    else:
        # Returns an array of sequences indices for that Viterbi should be
        # rerun with restrictions
        C = 2 * model_length
        C_state = state_seqs_max_lik == C
        prev_C_state = np.roll(C_state, 1, axis=2)
        prev_C_state[:,:,0] = False
        C_state_starts = C_state & ~prev_C_state
        previous_state = np.roll(state_seqs_max_lik, 1, axis=2).astype(np.int32)
        previous_state[:,:,0] = -1
        previous_is_match = (previous_state > 0) & (previous_state < model_length+1)
        #there are enough match states to align without repeat
        remaining_matches = model_length - previous_state
        remaining_residues = seq_lens[np.newaxis,:,np.newaxis] - np.arange(state_seqs_max_lik.shape[-1])[np.newaxis,np.newaxis,:]
        enough_matches = remaining_matches >= remaining_residues
        faulty_sequences = np.any(C_state_starts & previous_is_match & enough_matches, axis=-1)
        faulty_sequences = np.argwhere(faulty_sequences[0])[:,0]
        return faulty_sequences