import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
from packaging import version

from learnMSA.config.config import Configuration
import learnMSA.msa_hmm.AncProbsLayer as anc_probs
import learnMSA.msa_hmm.Emitter as emit
import learnMSA.msa_hmm.Priors as priors
import learnMSA.msa_hmm.Transitioner as trans
from learnMSA.model.tf.model import LearnMSAModel
from learnMSA.msa_hmm.alignment_metadata import AlignmentMetaData
from learnMSA.util.context import LearnMSAContext
from learnMSA.msa_hmm.AlignInsertions import AlignedInsertions
from learnMSA.model.tf.training import BatchGenerator
from learnMSA.util.aligned_dataset import AlignedDataset, SequenceDataset


class AlignmentModel():
    """
    Decodes alignments from a number of models, stores them in a memory
    friendly representation and generates table-form (memory unfriendly)
    alignments on demand (batch-wise mode possible).
    """

    data: SequenceDataset
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

    best_model: int
    """Index of the best model based on model selection criterion.
    TODO: Set externally, remove this attribute from here or set internally.
    """

    metadata: dict
    """Alignment metadata for each model."""

    def __init__(
        self,
        data: SequenceDataset,
        model: LearnMSAModel,
        indices: np.ndarray|None = None,
        gap_symbol: str = '-',
        gap_symbol_insertions: str = '.',
    ) -> None:
        """
        Args:
            data: The dataset of sequences.
            model: A learnMSA model instance for decoding and scoring.
            indices: An optional array of sequence indices specifying which
                sequences from data are included in the alignment. If None,
                all sequences are included.
            gap_symbol: Character used to denote missing match positions.
            gap_symbol_insertions: Character used to denote insertions in other
                sequences.
        """
        self.data = data
        self.model = model
        if indices is None:
            self.indices = np.arange(data.num_seq)
        else:
            self.indices = indices
        self.gap_symbol = gap_symbol
        self.gap_symbol_insertions = gap_symbol_insertions
        self.best_model = -1
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
                list(self.data.get_alphabet_no_gap()) +
                [self.gap_symbol] +
                list(self.data.get_alphabet_no_gap().lower()) +
                [self.gap_symbol_insertions, "$"]
            ))
        else:
            output_alphabet = np.array((
                list(self.data.get_alphabet_no_gap()) +
                [self.gap_symbol] +
                list(self.data.get_alphabet_no_gap()) +
                [self.gap_symbol, "$"]
            ))
        return output_alphabet


    def to_string(
        self,
        model_index: int | None = None,
        add_block_sep: bool = True,
        aligned_insertions: AlignedInsertions = AlignedInsertions(),
        a2m: bool = True,
        only_matches: bool = False,
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
        """
        if model_index is None:
            model_index = self.best_model
        output_alphabet = self.get_output_alphabet(a2m)
        batch_alignment = self.get_batch_alignment(
            model_index=model_index,
            batch_indices=np.arange(self.indices.size),
            add_block_sep=add_block_sep,
            aligned_insertions=aligned_insertions,
            only_matches=only_matches,
        )
        alignment_strings = self.batch_to_string(
            batch_alignment, output_alphabet=output_alphabet
        )
        return alignment_strings

    def to_file(
        self,
        filepath: str,
        model_index: int | None = None,
        batch_size: int = 100000,
        add_block_sep: bool = False,
        aligned_insertions : AlignedInsertions = AlignedInsertions(),
        format: str = "fasta",
        fasta_line_limit: int = 80,
        only_matches: bool = False,
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
        """
        if model_index is None:
            model_index = self.best_model
        if format == "fasta" or format == "a2m":
            # Stream batches to file
            output_alphabet = self.get_output_alphabet(format == "a2m")
            with open(filepath, "w") as output_file:
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
                    )
                    alignment_strings = self.batch_to_string(
                        batch_alignment, output_alphabet=output_alphabet
                    )
                    for s, seq_ind in zip(alignment_strings, batch_indices):
                        seq_header = self.data.get_header(
                            self.indices[seq_ind]
                        )
                        output_file.write(">"+seq_header+"\n")
                        for j in range(0, len(s), fasta_line_limit):
                            output_file.write(s[j:j+fasta_line_limit]+"\n")
                    i += batch_size
        else:
            # Decode the whole alignment into memory and write the entire
            # thing at once
            msa = self.to_string(
                model_index, add_block_sep, aligned_insertions
            )
            msa = [
                (self.data.seq_ids[self.indices[i]], msa[i])
                for i in range(len(msa))
            ]
            data = AlignedDataset(aligned_sequences=msa)
            data.write(filepath, format)

    def get_batch_alignment(
        self,
        model_index: int,
        batch_indices: np.ndarray,
        add_block_sep: bool = True,
        aligned_insertions: AlignedInsertions = AlignedInsertions(),
        only_matches: bool = False,
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
        """
        if not model_index in self.metadata:
            self._build_alignment([model_index])
        data = self.metadata[model_index]
        b = batch_indices.size
        sequences = np.zeros((b, self.data.max_len), dtype=np.uint16)
        sequences += (len(self.data.alphabet)-1)
        for i,j in enumerate(batch_indices):
            idx = int(self.indices[j])
            l = self.data.seq_lens[idx]
            sequences[i, :l] = self.data.get_encoded_seq(idx)
        blocks = []
        if add_block_sep:
            sep = np.zeros((b,1), dtype=np.uint16) + 2*len(self.data.alphabet)
        if not only_matches:
            left_flank_block = self.get_insertion_block(
                sequences,
                data.left_flank_len[batch_indices],
                max(data.left_flank_len_total, aligned_insertions.ext_left_flank),
                data.left_flank_start[batch_indices],
                adjust_to_right=True,
                custom_columns=aligned_insertions.left_flank(batch_indices)
            )
            blocks.append(left_flank_block)
            if add_block_sep:
                blocks.append(sep)
        for i in range(data.num_repeats):
            consensus = data.consensus[i]
            #remove columns consisting only of gaps
            is_non_empty = np.any(consensus != -1, axis=0)
            ins_len = data.insertion_lens[i]
            ins_start = data.insertion_start[i]
            alignment_block = self.get_alignment_block(
                sequences=sequences,
                consensus=consensus[batch_indices],
                ins_len=data.insertion_lens[i][batch_indices],
                ins_len_total=np.maximum(
                    data.insertion_lens_total,
                    aligned_insertions.ext_insertions
                )[i],
                ins_start=data.insertion_start[i][batch_indices],
                is_non_empty=is_non_empty,
                custom_columns=aligned_insertions.insertion(batch_indices, i),
                only_matches=only_matches
            )
            blocks.append(alignment_block)
            if add_block_sep:
                blocks.append(sep)
            if i < data.num_repeats-1 and not only_matches:
                unannotated_segment_l = data.unannotated_segments_len[i]
                unannotated_segment_s = data.unannotated_segments_start[i]
                unannotated_block = self.get_insertion_block(
                    sequences,
                    unannotated_segment_l[batch_indices],
                    np.maximum(
                        data.unannotated_segment_lens_total,
                        aligned_insertions.ext_unannotated
                    )[i],
                    unannotated_segment_s[batch_indices],
                    custom_columns=aligned_insertions.unannotated_segment(
                        batch_indices, i
                    )
                )
                blocks.append(unannotated_block)
                if add_block_sep:
                    blocks.append(sep)
        if not only_matches:
            right_flank_block = self.get_insertion_block(
                sequences,
                data.right_flank_len[batch_indices],
                max(data.right_flank_len_total, aligned_insertions.ext_right_flank),
                data.right_flank_start[batch_indices],
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
        alignment_arr = output_alphabet[batch_alignment]
        alignment_strings = [''.join(s) for s in alignment_arr]
        return alignment_strings

    def write_scores(self, filepath: Path, model: int|None = None) -> None:
        """ Writes per-sequence scores (loglik, bitscore) to a
            tsv file sorted by the bitscore ``loglik(S) - log P(S; nullmodel)``.
        Args:
            filepath: Path of the output file.
            model: The model for which scores are written. By default, the best
                model based on the model selection standard criterion is used.
        """
        # Find the model index to use
        if model is None:
            model = self.best_model if self.best_model >= 0 else 0  # type: ignore

        # Compute the likelihood and bitscores for all sequences
        loglik = self.model.estimate_loglik(
            self.data, self.data.num_seq, reduce=False, models=[model]
        )
        # Compute the bitscore
        log_null = self.model.compute_null_model_log_probs(self.data)
        bitscore = loglik - log_null

        # Sort by bitscore in descending order
        sorted_indices = np.argsort(-bitscore)

        # Write to file
        with open(filepath, "w") as scorefile:
            scorefile.write(
                "\t".join(["seq_id", "loglik", "bit_score"]) + "\n"
            )
            for idx in sorted_indices:
                scorefile.write("\t".join([
                    f"{self.data.seq_ids[idx]}",
                    f"{loglik[idx]}",
                    f"{bitscore[idx]}"
                ]) + "\n")

    def save(self, filepath: str, pack: bool = True) -> None:
        """ Writes the underlying models to file.

        Args:
            filepath: Path of the written file.
            pack: If true, the output will be a zip file, otherwise a directory.
        """
        Path(filepath).mkdir(parents=True, exist_ok=True)
        # Serialize metadata
        d: dict = {
            "gap_symbol" : self.gap_symbol,
            "gap_symbol_insertions" : self.gap_symbol_insertions,
        }
        if self.best_model >= 0:
            d["best_model"] = int(self.best_model)
        with open(filepath+"/meta.json", "w") as metafile:
            metafile.write(json.dumps(d, indent=4))
        # Serialize indices
        np.savetxt(filepath+"/indices", self.indices, fmt='%i')
        # Save the model
        self.model.save(filepath+".keras")
        if pack:
            shutil.make_archive(filepath, "zip", filepath)
            try:
                shutil.rmtree(filepath)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

    @classmethod
    def load(
        cls,
        filepath: str,
        data: SequenceDataset,
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
        if from_packed:
            shutil.unpack_archive(filepath+".zip", filepath)

        # Deserialize metadata
        with open(filepath+"/meta.json") as metafile:
            d = json.load(metafile)

        # Deserialize indices
        indices = np.loadtxt(filepath+"/indices", dtype=int)

        # Load the model
        with warnings.catch_warnings():
            # Suppress the compile warning since we manually compile right after
            warnings.filterwarnings(
                'ignore',
                message=".*compile.*was not called as part of model loading.*",
                category=UserWarning
            )
            model = tf.keras.models.load_model(
                filepath+".keras",
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
        )
        if "best_model" in d:
            am.best_model = d["best_model"]
        return am


    @classmethod
    def decode_core(cls, model_length, state_seqs_max_lik, indices):
        """ Decodes consensus columns as a matrix as well as insertion lengths
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
        n = state_seqs_max_lik.shape[0]
        c = model_length
        # initialize the consensus with gaps
        consensus_columns = -np.ones((n, c), dtype=np.int16)
        # insertion lengths and starting positions per sequence
        insertion_lens = np.zeros((n, c-1), dtype=np.int16)
        insertion_start = -np.ones((n, c-1), dtype=np.int16)
        # is true if and only if the previous hidden state was an insertion
        # state (not counting flanks)
        last_insert = np.zeros(n, dtype=bool)
        A = np.arange(n)
        while True:
            q = state_seqs_max_lik[A, indices]
            is_match = ((q > 0) & (q < c+1))
            is_insert = ((q >= c+1) & (q < 2*c))
            is_insert_start = is_insert & ~last_insert
            is_unannotated = (q == 2*c)
            is_at_end = ((q == 2*c+1) | (q == 2*c+2))
            if np.all(is_unannotated | is_at_end):
                finished = ~is_unannotated
                break
            # track matches
            consensus_columns[A[is_match], q[is_match]-1] = indices[is_match]
            # track insertions
            is_insert_subset = A[is_insert]
            is_insert_start_subset = A[is_insert_start]
            insertion_lens[is_insert_subset, q[is_insert]-c-1] += 1
            insertion_start[is_insert_start_subset, q[is_insert_start]-c-1] = indices[is_insert_start]
            indices[is_match | is_insert] += 1
            last_insert = is_insert
        return consensus_columns, insertion_lens, insertion_start, finished


    @classmethod
    def decode_flank(cls, state_seqs_max_lik, flank_state_id, indices):
        """ Decodes flanking insertion states. The deconding is active as long
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
        n = state_seqs_max_lik.shape[0]
        insertion_start = np.copy(indices)
        while True:
            q = state_seqs_max_lik[np.arange(n), indices]
            is_flank = (q == flank_state_id)
            if ~np.any(is_flank):
                break
            indices[is_flank] += 1
        insertion_lens = indices - insertion_start
        return insertion_lens, insertion_start


    @classmethod
    def decode(cls, model_length, state_seqs_max_lik):
        """ Decodes an implicit alignment (insertion start/length are
            represented as 2 integers) from most likely state sequences.
        Args:
            model_length: Number of match states (length of the consensus
                sequence).
            state_seqs_max_lik: A tensor with the most likeli state sequences.
                Shape: (num_seq, L)
        Returns:
            core_blocks: Representation of the consensus.
            left_flank:
            right_flank:
            unannotated_segments:
        """
        n = state_seqs_max_lik.shape[0]
        c = model_length #alias for code readability
        indices = np.zeros(n, np.int16) # active positions in the sequence
        left_flank = cls.decode_flank(state_seqs_max_lik, 0, indices)
        core_blocks = []
        unannotated_segments = []
        while True:
            C, IL, IS, finished = cls.decode_core(
                model_length, state_seqs_max_lik, indices
            )
            core_blocks.append((C, IL, IS, finished))
            if np.all(finished):
                break
            unannotated_segments.append(
                cls.decode_flank(state_seqs_max_lik, 2*c, indices)
            )
        right_flank = cls.decode_flank(state_seqs_max_lik, 2*c+1, indices)
        return core_blocks, left_flank, right_flank, unannotated_segments


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
        only_matches=False
    ):
        """ Constructs one core model hit block from an implicitly represented
            alignment.

        Args:

        Returns:
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
    def _build_alignment(self, models):

        assert len(models) == 1, "Not implemented for multiple models."

        self.model.viterbi_mode()
        self.model.compile()
        state_seqs_max_lik = self.model.predict(
            self.data, self.indices, models
        )

        # TODO: transpose needed to make legacy code work, fix later
        state_seqs_max_lik = np.transpose(state_seqs_max_lik, (2, 0, 1)) # (num_model, num_seq, L)
        state_seqs_max_lik[state_seqs_max_lik == -1] = 2*self.model.context.model_lengths[0]+2 # terminal state

        # TODO: the legacy code assumes a different indexing of the pHMM states
        # legacy: 0: left flank, 1..C: match states, C+1..2C-1: insert states,
        # 2C: right flank, 2C+1: unannotated, -1: terminal state
        # current: 0..C-1: match states, C..2C-2: insert states,
        # 2C-1: left flank, 2C: right flank, 2C+1: unannotated, 2C+2: terminal state

        # Translate from current indexing to legacy indexing
        for model_idx in models:
            C = self.model.context.model_lengths[model_idx]
            states = state_seqs_max_lik[models.index(model_idx)]

            # Create a copy to avoid in-place modification issues
            translated_states = np.copy(states)

            # Left flank: 2C-1 → 0
            translated_states[states == 2*C-1] = 0

            # Match and insert states: [0, 2C-1) → [1, 2C)
            mask = states < 2*C-1
            translated_states[mask] = states[mask] + 1

            # Right flank, unannotated, terminal: >= 2C stay the same
            # (already copied, no change needed)

            state_seqs_max_lik[models.index(model_idx)] = translated_states

        state_seqs_max_lik = self._clean_up_viterbi_seqs(
            state_seqs_max_lik, models
        )
        for i,max_lik_seqs in zip(models, state_seqs_max_lik):
            model_len = self.model.context.model_lengths[i]
            decoded_data = AlignmentModel.decode(model_len, max_lik_seqs)
            self.metadata[i] = AlignmentMetaData(*decoded_data)

    def _clean_up_viterbi_seqs(self, state_seqs_max_lik, models):

        assert len(models) == 1, "Not implemented for multiple models."

        # TODO
        print("WARNING: Fixing faulty Viterbi sequences is currently not supported. SKIPPING.")
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