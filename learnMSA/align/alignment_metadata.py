from dataclasses import dataclass, field
import numpy as np


# utility class used in AlignmentModel storing data to construct a full MSA
@dataclass
class AlignmentMetaData:

    num_rows: int
    """Number of sequences in the alignment."""

    num_match: int
    """Total number of match states in pHMM."""

    num_repeats_per_row: np.ndarray
    """An array of shape (num_rows,) with the number of actual (non-empty)
    repeats for each row.
    """

    domain_hit: np.ndarray
    """An array of shape (total_repeats, num_match) with the domain hits,
    where total_repeats = sum(num_repeats_per_row). Stored in row-major flat
    order: all repeats for row 0, then row 1, etc.
    """

    domain_loc: np.ndarray
    """An array of shape (total_repeats, 2) with the starting and ending
    positions of the domain hits in the sequences.
    """

    insertion_lens: np.ndarray
    """An array of shape (total_repeats, num_match - 1) with the lengths of
    insertions.
    """

    insertion_start: np.ndarray
    """An array of shape (total_repeats, num_match - 1) with the starting
    positions of insertions.
    """

    left_flank_len: np.ndarray
    """An array of shape (num_rows,) containing the lengths of the left flanks.
    """

    left_flank_start: np.ndarray
    """An array of shape (num_rows,) containing the starting positions of the
    left flanks.
    """

    right_flank_len: np.ndarray
    """An array of shape (num_rows,) containing the lengths of the right flanks.
    """

    right_flank_start: np.ndarray
    """An array of shape (num_rows,) containing the starting positions of the
    right flanks.
    """

    unannotated_segments_len: np.ndarray
    """A flat array of shape (total_unannotated,) where
    total_unannotated = sum(max(0, num_repeats_per_row - 1)).
    Stores the lengths of unannotated segments between consecutive repeats,
    in row-major order: for row j all (num_repeats_per_row[j]-1) segments.
    """

    unannotated_segments_start: np.ndarray
    """A flat array of shape (total_unannotated,) with starting positions of
    unannotated segments (same layout as unannotated_segments_len).
    """

    sort_perm: np.ndarray | None = field(default=None, repr=False)
    """Optional permutation array of shape (num_rows,) set when data is stored
    in sorted order (e.g. by :func:`~learnMSA.align.tf.decode.decode_tf`).
    ``sort_perm[k]`` is the *original* row index of the k-th sorted row.
    When present, ``_inv_perm`` is computed in ``__post_init__`` and all
    accessor methods (``get_repeat_data``, ``get_unannotated_data``, and the
    ``*_for`` getters) transparently map original row indices to storage
    positions.
    """

    def __post_init__(self):
        nrpr = np.asarray(self.num_repeats_per_row, dtype=np.int32)
        self.num_repeats_per_row = nrpr
        # _row_offsets[j] = index into domain_hit of row j's first repeat
        self._row_offsets = np.concatenate(
            [[0], np.cumsum(nrpr)]
        ).astype(np.int32)
        # _repeat_offset[j] = virtual (logical) column of row j's first repeat
        # (all zeros initially; modified by shift())
        self._repeat_offset = np.zeros(self.num_rows, dtype=np.int32)
        # Build inverse permutation if sort_perm is provided.
        if self.sort_perm is not None:
            perm = np.asarray(self.sort_perm, dtype=np.int32)
            self.sort_perm = perm
            inv = np.empty_like(perm)
            inv[perm] = np.arange(len(perm), dtype=np.int32)
            self._inv_perm: np.ndarray | None = inv
        else:
            self._inv_perm = None

    # ------------------------------------------------------------------
    # Derived scalar / array properties
    # ------------------------------------------------------------------

    @property
    def num_repeats(self) -> int:
        """Maximum virtual repeat index + 1 (= alignment width in repeats)."""
        if self.num_rows == 0:
            return 0
        return int(np.amax(self._repeat_offset + self.num_repeats_per_row))

    @property
    def skip(self) -> np.ndarray:
        """Compatibility view: (num_repeats, num_rows) bool.
        skip[r, j] = False iff row j has an active (non-last) repeat at
        virtual position r, matching the old semantics where False = 'not yet
        finished' and True = 'finished / padding'.
        Specifically: skip[r, j] = False when
            _repeat_offset[j] <= r < _repeat_offset[j] + num_repeats_per_row[j] - 1
        and True otherwise.
        """
        R = self.num_repeats
        N = self.num_rows
        result = np.ones((R, N), dtype=bool)
        if R == 0 or N == 0:
            return result
        r_idx = np.arange(R, dtype=np.int32)[:, np.newaxis]         # (R, 1)
        start = self._repeat_offset[np.newaxis, :]                   # (1, N)
        end   = (start + self.num_repeats_per_row[np.newaxis, :] - 1)  # (1, N)
        # is_active: row j has a non-last actual repeat at position r
        is_active = (r_idx >= start) & (r_idx < end)
        result[is_active] = False
        if self._inv_perm is not None:
            # Return columns in original-row order so that callers can index
            # with original sequence indices without knowing sort_perm.
            return result[:, self._inv_perm]
        return result

    # ------------------------------------------------------------------
    # Row-index translation (sort_perm support)
    # ------------------------------------------------------------------

    def _resolve_rows(self, row_indices: np.ndarray) -> np.ndarray:
        """Map *original* row indices to storage (sorted) row indices.

        When ``sort_perm`` is set the data arrays are stored in sorted order,
        so we must translate caller-supplied original indices before indexing
        into ``_row_offsets`` / ``num_repeats_per_row`` etc.
        """
        if self._inv_perm is None:
            return row_indices
        return self._inv_perm[row_indices]

    # ------------------------------------------------------------------
    # Core flat-to-virtual helpers
    # ------------------------------------------------------------------

    def _flat_virt_rep_and_row(self) -> tuple:
        """Return (virtual_repeat_idx, row_idx) arrays for every flat entry."""
        total_R = len(self.domain_hit)
        flat_to_row = np.searchsorted(
            self._row_offsets[1:],
            np.arange(total_R, dtype=np.int32),
            side='right',
        ).astype(np.int32)
        flat_local = (
            np.arange(total_R, dtype=np.int32) - self._row_offsets[flat_to_row]
        )
        virt_rep = (self._repeat_offset[flat_to_row] + flat_local).astype(np.int32)
        return virt_rep, flat_to_row

    # ------------------------------------------------------------------
    # Per-repeat data access
    # ------------------------------------------------------------------

    def get_repeat_data(
        self, repeat_idx: int, row_indices: np.ndarray
    ) -> tuple:
        """Return per-repeat arrays for *row_indices* at virtual repeat *repeat_idx*.

        Rows that don't have this virtual repeat (outside their local range)
        receive padding: -1 for domain_hit / insertion_start / domain_loc,
        0 for insertion_lens.

        Args:
            repeat_idx: Virtual (logical) repeat column index.
            row_indices: 1-D int array of row indices to query.

        Returns:
            domain_hit_slice   : (B, num_match)       int16
            insertion_lens_slice: (B, num_match-1)    int16
            insertion_start_slice: (B, num_match-1)   int16
            domain_loc_slice   : (B, 2)
            has_repeat         : (B,) bool
        """
        sr = self._resolve_rows(row_indices)
        local_idx = repeat_idx - self._repeat_offset[sr]
        has_repeat = (
            (local_idx >= 0)
            & (local_idx < self.num_repeats_per_row[sr])
        )
        safe_local = np.clip(
            local_idx, 0,
            np.maximum(self.num_repeats_per_row[sr] - 1, 0),
        )
        flat_idx = np.clip(
            self._row_offsets[sr] + safe_local,
            0, max(0, len(self.domain_hit) - 1),
        )
        dh = self.domain_hit[flat_idx].copy()
        dh[~has_repeat] = -1
        il = self.insertion_lens[flat_idx].copy()
        il[~has_repeat] = 0
        is_ = self.insertion_start[flat_idx].copy()
        is_[~has_repeat] = -1
        dl = self.domain_loc[flat_idx].copy()
        dl[~has_repeat] = -1
        return dh, il, is_, dl, has_repeat

    def get_unannotated_data(
        self, repeat_idx: int, row_indices: np.ndarray
    ) -> tuple:
        """Return (len, start) of the unannotated segment *after* virtual
        repeat *repeat_idx* for *row_indices*.

        Rows without this unannotated segment return (0, -1).
        """
        sr = self._resolve_rows(row_indices)
        local_r = repeat_idx - self._repeat_offset[sr]
        has_seg = (
            (local_r >= 0)
            & (local_r < self.num_repeats_per_row[sr] - 1)
        )
        safe_local = np.clip(
            local_r, 0,
            np.maximum(self.num_repeats_per_row[sr] - 2, 0),
        )
        uns_row_off = (
            self._row_offsets[:-1] - np.arange(self.num_rows, dtype=np.int32)
        )
        flat_idx = np.clip(
            uns_row_off[sr] + safe_local,
            0, max(0, len(self.unannotated_segments_len) - 1),
        )
        l = self.unannotated_segments_len[flat_idx].copy()
        l[~has_seg] = 0
        s = self.unannotated_segments_start[flat_idx].copy()
        s[~has_seg] = -1
        return l, s

    # ------------------------------------------------------------------
    # Per-row flank getters (respect sort_perm)
    # ------------------------------------------------------------------

    def left_flank_len_for(self, row_indices: np.ndarray) -> np.ndarray:
        """Return ``left_flank_len`` for *row_indices* (original ordering)."""
        return self.left_flank_len[self._resolve_rows(row_indices)]

    def left_flank_start_for(self, row_indices: np.ndarray) -> np.ndarray:
        """Return ``left_flank_start`` for *row_indices* (original ordering)."""
        return self.left_flank_start[self._resolve_rows(row_indices)]

    def right_flank_len_for(self, row_indices: np.ndarray) -> np.ndarray:
        """Return ``right_flank_len`` for *row_indices* (original ordering)."""
        return self.right_flank_len[self._resolve_rows(row_indices)]

    def right_flank_start_for(self, row_indices: np.ndarray) -> np.ndarray:
        """Return ``right_flank_start`` for *row_indices* (original ordering)."""
        return self.right_flank_start[self._resolve_rows(row_indices)]

    # ------------------------------------------------------------------
    # Aggregate properties
    # ------------------------------------------------------------------

    def occupancy_matrix(self) -> np.ndarray:
        """Return (num_repeats, num_rows) int32 with the number of used match
        states per (virtual repeat, row). Empty slots are -1.
        """
        R = self.num_repeats
        N = self.num_rows
        result = np.full((R, N), -1, dtype=np.int32)
        total_R = len(self.domain_hit)
        if total_R == 0:
            return result
        virt_rep, flat_to_row = self._flat_virt_rep_and_row()
        occ = np.sum(self.domain_hit != -1, axis=-1).astype(np.int32)
        occ[occ == 0] = -1
        result[virt_rep, flat_to_row] = occ
        return result

    def repeat_occupancy_mask(self) -> np.ndarray:
        """Return (num_repeats, num_match) bool, True if any row has data."""
        R = self.num_repeats
        M = self.num_match
        result = np.zeros((R, M), dtype=np.int8)
        total_R = len(self.domain_hit)
        if total_R > 0:
            virt_rep, _ = self._flat_virt_rep_and_row()
            non_gap = (self.domain_hit != -1).astype(np.int8)
            np.maximum.at(result, virt_rep, non_gap)
        return result.astype(bool)

    @property
    def insertion_lens_total(self) -> np.ndarray:
        """(num_repeats, num_match-1) int32: max insertion length per slot."""
        R = self.num_repeats
        M = self.num_match
        result = np.zeros(R * max(0, M - 1), dtype=np.int32)
        total_R = len(self.insertion_lens)
        if total_R == 0 or M <= 1:
            return result.reshape(R, max(0, M - 1))
        virt_rep, _ = self._flat_virt_rep_and_row()
        ins_flat = self.insertion_lens.astype(np.int32).ravel()
        lin_idx = (
            virt_rep[:, np.newaxis] * (M - 1)
            + np.arange(M - 1, dtype=np.int32)[np.newaxis, :]
        ).ravel()
        np.maximum.at(result, lin_idx, ins_flat)
        return result.reshape(R, M - 1)

    @property
    def left_flank_len_total(self) -> np.int32:
        """Maximum left-flank length across all sequences."""
        return np.amax(self.left_flank_len).astype(np.int32)

    @property
    def right_flank_len_total(self) -> np.int32:
        """Maximum right-flank length across all sequences."""
        return np.amax(self.right_flank_len).astype(np.int32)

    @property
    def unannotated_segment_lens_total(self) -> np.ndarray:
        """(num_repeats-1,) int32: max unannotated segment length per gap."""
        R = max(0, self.num_repeats - 1)
        result = np.zeros(R, dtype=np.int32)
        total_U = len(self.unannotated_segments_len)
        if total_U == 0:
            return result
        uns_row_off = (
            self._row_offsets[:-1] - np.arange(self.num_rows, dtype=np.int32)
        )
        uns_row_off_ext = np.concatenate([uns_row_off, [total_U]]).astype(np.int32)
        flat_to_row = np.searchsorted(
            uns_row_off_ext[1:],
            np.arange(total_U, dtype=np.int32),
            side='right',
        ).astype(np.int32)
        flat_local = (
            np.arange(total_U, dtype=np.int32) - uns_row_off[flat_to_row]
        )
        virt_rep = (self._repeat_offset[flat_to_row] + flat_local).astype(np.int32)
        np.maximum.at(result, virt_rep, self.unannotated_segments_len)
        return result

    @property
    def alignment_len(self) -> int:
        """Total alignment length including flanks, core blocks, insertions,
        and unannotated segments. May include empty columns removed later.
        """
        return (
            self.left_flank_len_total
            + self.num_match * self.num_repeats
            + np.sum(self.insertion_lens_total)
            + np.sum(self.unannotated_segment_lens_total)
            + self.right_flank_len_total
        )

    # ------------------------------------------------------------------
    # Shift (logical re-alignment of domain hits)
    # ------------------------------------------------------------------

    def shift(self, shift: np.ndarray) -> None:
        """Logically shift each row's domain hits right by *shift[j]* repeat
        positions.  Only the virtual-column offset is updated; the flat data
        arrays are unchanged.

        Args:
            shift: (num_rows,) int array of non-negative shift values.
        """
        self._repeat_offset += shift.astype(np.int32)
