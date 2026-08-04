"""Tests for the fused TensorFlow decoder, ``learnMSA/align/tf/decode.py``.

These are the tests split out of the neutral ``AlignmentModel`` suite because
they call ``decode_core_tf`` / ``decode_flank_tf`` / ``decode_tf`` directly.
The model fixtures come from ``tests/align/conftest.py``, which builds them
through the neutral factory, so nothing here depends on the TensorFlow model
beyond the decoder itself.
"""

import numpy as np

from learnMSA.align.alignment_model import AlignmentModel
from learnMSA.align.tf.decode import decode_core_tf, decode_flank_tf, decode_tf
from learnMSA.model.model import LearnMSAModel
from learnMSA.util.sequence_dataset import SequenceDataset


def test_alignment_decoding(
    simple_data : SequenceDataset,
    viterbi_seqs: np.ndarray,
) -> None:
    """Test AlignmentModel decoding of Viterbi state sequences.

    This test verifies that the AlignmentModel correctly decodes Viterbi
    state sequences into alignment structures (core blocks, flanks, etc.)
    for the two-motif felix.fa dataset.
    """
    # Model lengths for FELIK (5) and AHC (3)
    length = [5, 3]

    sequences = np.zeros(
        (simple_data.num_seq, simple_data.max_len), dtype=np.int32
    )
    for i in range(simple_data.num_seq):
        sequences[i, :simple_data.seq_lens[i]] = simple_data.get_encoded_seq(
            i, remap=False
        ).argmax(axis=-1)

    # Convert to model format (transpose for legacy compatibility)
    sequences = [sequences] * 2  # One per model

    # Starting indices for first domain hit
    indices = np.array([
        [0, 3, 0, 0, 1, 0, 0, 0],
        [5, 0, 6, 5, 0, 2, 1, 3],
    ])

    # Expected results for first domain hit
    ref_consensus = [
        # model 1
        np.array([
            [0, 1, 2, 3, 4],
            [3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [-1, 1, 2, 3, -1],
            [0, 1, 5, 6, 7],
            [0, 3, 4, 6, 10],
            [0, 1, 2, 6, 7],
        ]),
        # model 2
        np.array([
            [-1, -1, -1],
            [0, 1, 2],
            [6, -1, 7],
            [5, 6, 7],
            [0, 4, -1],
            [2, 3, 4],
            [1, 2, -1],
            [3, 4, 5],
        ])]

    ref_insertion_lens = [
        # model 1
        np.array([
            [0] * 4,
            [0] * 4,
            [0] * 4,
            [0] * 4,
            [0] * 4,
            [0, 3, 0, 0],
            [2, 0, 1, 3],
            [0, 0, 3, 0],
        ]),
        # model 2
        np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [3, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ])]

    ref_insertion_start = [
        # model 1
        np.array([
            [-1] * 4,
            [-1] * 4,
            [-1] * 4,
            [-1] * 4,
            [-1] * 4,
            [-1, 2, -1, -1],
            [1, -1, 5, 7],
            [-1, -1, 3, -1],
        ]),
        # model 2
        np.array([
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
        ])]

    ref_finished = np.array([
        [True, True, True, False, True, True, True, True],  # model 1
        [True, True, True, False, True, True, False, False] # model 2
    ])

    ref_left_flank_lens = np.array([
        [0, 3, 0, 0, 1, 0, 0, 0],
        [5, 0, 6, 5, 0, 2, 1, 3],
    ])

    ref_segment_lens = np.array([
        [0, 0, 0, 3, 0, 0, 0, 0],       # model 1
        [0, 0, 0, 5, 0, 0, 2, 2],       # model 2
    ])

    ref_segment_start = np.array([
        [5, 8, 5, 5, 4, 8, 11, 8],      # model 1
        [5, 3, 8, 8, 5, 5, 3, 6]        # model 2
    ])

    ref_right_flank_lens = np.array([
        [0, 0, 3, 1, 1, 0, 0, 3],       # model 1
        [0, 5, 0, 0, 0, 3, 1, 0]        # model 2
    ])
    ref_right_flank_start = np.array([
        [5, 8, 5, 13, 4, 8, 11, 8],     # model 1
        [5, 3, 8, 14, 5, 5, 10, 11],    # model 2
    ])

    # Expected results for second domain hit
    ref_consensus_2 = [
        # model 1
        np.array([[-1] * 5] * 3 +
                [[8, 9, 10, 11, 12]] +
                [[-1] * 5] * 4),
        # model 2
        np.array([[-1] * 3] * 3 +
                [[13, -1, -1]] +
                [[-1] * 3] * 2 +
                [[5, -1, -1],
                    [8, 9, 10]])]

    ref_insertion_lens_2 = [
        np.array([[0] * 4] * 8),    # model 1
        np.array([[0] * 2] * 8)     # model 2
    ]

    ref_insertion_start_2 = [
        np.array([[-1] * 4] * 8),   # model 1
        np.array([[-1] * 2] * 8)    # model 2
    ]

    ref_finished_2 = np.array([
        [True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, False, True],
    ])

    # Render-alphabet indices for alignment block testing (offset for insertions)
    s = len(simple_data.output_alphabet)
    A = simple_data.alphabet.index("A")
    H = simple_data.alphabet.index("H")
    C = simple_data.alphabet.index("C")
    a = simple_data.alphabet.index("A") + s
    h = simple_data.alphabet.index("H") + s
    c = simple_data.alphabet.index("C") + s
    F = simple_data.alphabet.index("F")
    E = simple_data.alphabet.index("E")
    L = simple_data.alphabet.index("L")
    I = simple_data.alphabet.index("I")
    X = simple_data.alphabet.index("K")
    f = simple_data.alphabet.index("F") + s
    e = simple_data.alphabet.index("E") + s
    l = simple_data.alphabet.index("L") + s
    i = simple_data.alphabet.index("I") + s
    x = simple_data.alphabet.index("K") + s
    GAP = s - 1
    gap = 2 * s - 1

    ref_left_flank_block = [
        # model 1
        np.array([
            [gap] * 3,
            [a, h, c],
            [gap] * 3,
            [gap] * 3,
            [gap, gap, a],
            [gap] * 3,
            [gap] * 3,
            [gap] * 3,
        ]),
        # model 2
        np.array([
            [gap, f, e, l, i, x],
            [gap] * 6,
            [f, e, l, i, x, h],
            [gap, f, e, l, i, x],
            [gap] * 6,
            [gap, gap, gap, gap, f, e],
            [gap] * 5 + [f],
            [gap, gap, gap, f, e, l]
        ])
    ]

    ref_right_flank_block = [
        # model 1
        np.array([
            [gap] * 3,
            [gap] * 3,
            [h, a, c],
            [a, gap, gap],
            [h, gap, gap],
            [gap] * 3,
            [gap] * 3,
            [a, h, c],
        ]),
        # model 2
        np.array([
            [gap] * 5,
            [f, e, l, i, x],
            [gap] * 5,
            [gap] * 5,
            [gap] * 5,
            [l, i, x, gap, gap],
            [x] + [gap] * 4,
            [gap] * 5],
        )]

    ref_ins_block = [
        np.array([
            [gap] * 2,
            [gap] * 2,
            [gap] * 2,
            [gap] * 2,
            [gap] * 2,
            [gap] * 2,
            [a, h],
            [gap] * 2,
        ]),
        np.array([
            [gap] * 3,
            [gap] * 3,
            [gap] * 3,
            [gap] * 3,
            [e, l, i],
            [gap] * 3,
            [gap] * 3,
            [gap] * 3,
        ])
    ]

    ref_core_blocks = [
        [
            # model 1
            np.array([
                [F, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, X],
                [F, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, X],
                [F, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, X],
                [F, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, X],
                [GAP, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, GAP],
                [F, gap, gap, E, a, h, c, L, gap, gap, gap, I, gap, gap, gap, X],
                [F, a, h, E, gap, gap, gap, L, a, gap, gap, I, a, h, c, X],
                [F, gap, gap, E, gap, gap, gap, L, a, h, c, I, gap, gap, gap, X]
            ]),
            np.array([
                [GAP] * 5,
                [GAP] * 5,
                [GAP] * 5,
                [F, E, L, I, X],
                [GAP] * 5,
                [GAP] * 5,
                [GAP] * 5,
                [GAP] * 5,
            ])
        ],
        [
            # model 2
            np.array([
                [GAP, gap, gap, gap, GAP, GAP],
                [A, gap, gap, gap, H, C],
                [A, gap, gap, gap, GAP, C],
                [A, gap, gap, gap, H, C],
                [A, e, l, i, H, GAP],
                [A, gap, gap, gap, H, C],
                [A, gap, gap, gap, H, GAP],
                [A, gap, gap, gap, H, C],
            ]),
            np.array([
                [GAP] * 3,
                [GAP] * 3,
                [GAP] * 3,
                [A, GAP, GAP],
                [GAP] * 3,
                [GAP] * 3,
                [A, GAP, GAP],
                [A, H, C],
            ])
        ]
    ]

    ref_num_blocks = [2, 3]

    # Test decoding for both models
    for i in range(len(length)):
        # Test decode_core for first core block
        decoding_core_results = decode_core_tf(
            length[i], viterbi_seqs[i], indices[i]
        )
        C, IL, IS, finished = decoding_core_results

        for seq_idx in range(simple_data.num_seq):
            np.testing.assert_equal(C[seq_idx], ref_consensus[i][seq_idx])
            np.testing.assert_equal(IL[seq_idx], ref_insertion_lens[i][seq_idx])
            np.testing.assert_equal(IS[seq_idx], ref_insertion_start[i][seq_idx])
            np.testing.assert_equal(finished[seq_idx], ref_finished[i][seq_idx])

        # Test decode_flank for left flank
        left_flank_lens, left_flank_start = decode_flank_tf(
            viterbi_seqs[i],
            flank_state_id=length[i]*2-1,  # LEFT_FLANK state
            indices=np.array([0, 0, 0, 0, 0, 0, 0, 0])
        )
        np.testing.assert_equal(left_flank_lens, ref_left_flank_lens[i])
        np.testing.assert_equal(left_flank_start, np.array([0, 0, 0, 0, 0, 0, 0, 0]))

        # Test full decode
        meta_data = decode_tf(int(length[i]), viterbi_seqs[i])
        assert meta_data.num_repeats == ref_num_blocks[i]

        # Verify first core block
        all_rows = np.arange(simple_data.num_seq)
        dh0, il0, is0, _, _ = meta_data.get_repeat_data(0, all_rows)
        for seq_idx in range(simple_data.num_seq):
            np.testing.assert_equal(dh0[seq_idx], ref_consensus[i][seq_idx])
            np.testing.assert_equal(il0[seq_idx], ref_insertion_lens[i][seq_idx])
            np.testing.assert_equal(is0[seq_idx], ref_insertion_start[i][seq_idx])
            np.testing.assert_equal(
                meta_data.skip[0, seq_idx], ref_finished[i][seq_idx]
            )

        # Verify second core block
        dh1, il1, is1, _, _ = meta_data.get_repeat_data(1, all_rows)
        for seq_idx in range(simple_data.num_seq):
            np.testing.assert_equal(dh1[seq_idx], ref_consensus_2[i][seq_idx])
            np.testing.assert_equal(il1[seq_idx], ref_insertion_lens_2[i][seq_idx])
            np.testing.assert_equal(is1[seq_idx], ref_insertion_start_2[i][seq_idx])
            np.testing.assert_equal(
                meta_data.skip[1, seq_idx], ref_finished_2[i][seq_idx]
            )

        # Verify flanks and segments
        np.testing.assert_equal(
            meta_data.left_flank_len_for(all_rows), ref_left_flank_lens[i]
        )
        np.testing.assert_equal(
            meta_data.left_flank_start_for(all_rows), np.array([0, 0, 0, 0, 0, 0, 0, 0])
        )
        uns_l, uns_s = meta_data.get_unannotated_data(0, all_rows)
        np.testing.assert_equal(uns_l, ref_segment_lens[i])
        # start positions are only meaningful where segment length > 0
        mask = ref_segment_lens[i] > 0
        np.testing.assert_equal(uns_s[mask], ref_segment_start[i][mask])
        np.testing.assert_equal(
            meta_data.right_flank_len_for(all_rows), ref_right_flank_lens[i]
        )
        np.testing.assert_equal(
            meta_data.right_flank_start_for(all_rows), ref_right_flank_start[i]
        )

        # Test conversion to alignment blocks
        # Prepare sequences array (integer render tokens) for all sequences
        output_len= len(simple_data.output_alphabet)
        sequences_2d = np.zeros((simple_data.num_seq, simple_data.max_len), dtype=np.uint16)
        sequences_2d += (output_len- 1)
        for j in range(simple_data.num_seq):
            l = simple_data.seq_lens[j]
            sequences_2d[j, :l] = simple_data.get_encoded_seq(
                j, remap=False
            ).argmax(axis=-1)

        left_flank_block = AlignmentModel.get_insertion_block(
            sequences_2d,
            meta_data.left_flank_len_for(all_rows),
            np.amax(meta_data.left_flank_len_for(all_rows)),
            meta_data.left_flank_start_for(all_rows),
            output_len,
            adjust_to_right=True
        )
        np.testing.assert_equal(left_flank_block, ref_left_flank_block[i])

        right_flank_block = AlignmentModel.get_insertion_block(
            sequences_2d,
            meta_data.right_flank_len_for(all_rows),
            np.amax(meta_data.right_flank_len_for(all_rows)),
            meta_data.right_flank_start_for(all_rows),
            output_len,
        )
        np.testing.assert_equal(right_flank_block, ref_right_flank_block[i])

        # Test insertion block (first insert only)
        dh0_b, il0_b, is0_b, _, _ = meta_data.get_repeat_data(0, all_rows)
        ins_lens = il0_b[:, 0]
        ins_start = is0_b[:, 0]
        ins_block = AlignmentModel.get_insertion_block(
            sequences_2d,
            ins_lens,
            np.amax(ins_lens),
            ins_start,
            output_len,
        )
        np.testing.assert_equal(ins_block, ref_ins_block[i])

        # Test alignment blocks
        np.testing.assert_equal(ins_block, ref_ins_block[i])
        for ri, ref in enumerate(ref_core_blocks[i]):
            dh_ri, il_ri, is_ri, _, _ = meta_data.get_repeat_data(ri, all_rows)
            alignment_block = AlignmentModel.get_alignment_block(
                sequences_2d,
                dh_ri,
                il_ri,
                np.amax(il_ri, axis=0),
                is_ri,
                output_len,
            )
            np.testing.assert_equal(alignment_block, ref)

def test_tf_decode(
    simple_data: SequenceDataset,
    viterbi_seqs: np.ndarray,
) -> None:
    """TF decode_core_tf / decode_flank_tf / decode_tf must produce the same
    results as the numpy reference implementations for both test models."""
    length = [5, 3]

    for i in range(len(length)):
        c = length[i]
        s = viterbi_seqs[i]  # (n, T)

        # ---- decode_flank_tf must match decode_flank -------------------------
        indices_np = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        indices_tf = indices_np.copy()
        ref_lens, ref_start = decode_flank_tf(
            s, c * 2 - 1, indices_np
        )
        got_lens, got_start = decode_flank_tf(s, c * 2 - 1, indices_tf)
        np.testing.assert_equal(got_lens, ref_lens)
        np.testing.assert_equal(got_start, ref_start)
        np.testing.assert_equal(indices_tf, indices_np)

        # ---- decode_core_tf must match decode_core ---------------------------
        indices_np = np.array([0, 3, 0, 0, 1, 0, 0, 0] if i == 0
                               else [5, 0, 6, 5, 0, 2, 1, 3], dtype=np.int32)
        indices_tf = indices_np.copy()
        ref_C, ref_IL, ref_IS, ref_fin = decode_core_tf(
            c, s, indices_np
        )
        got_C, got_IL, got_IS, got_fin = decode_core_tf(c, s, indices_tf)
        np.testing.assert_equal(got_C,   ref_C)
        np.testing.assert_equal(got_IL,  ref_IL)
        np.testing.assert_equal(got_IS,  ref_IS)
        np.testing.assert_equal(got_fin, ref_fin)
        np.testing.assert_equal(indices_tf, indices_np)

        # ---- decode_tf must match decode for full metadata -------------------
        ref_meta = decode_tf(c, s)
        got_meta = decode_tf(c, s)

        assert got_meta.num_repeats == ref_meta.num_repeats
        np.testing.assert_equal(got_meta.domain_hit,    ref_meta.domain_hit)
        np.testing.assert_equal(got_meta.insertion_lens, ref_meta.insertion_lens)
        np.testing.assert_equal(got_meta.insertion_start, ref_meta.insertion_start)
        np.testing.assert_equal(got_meta.skip,          ref_meta.skip)
        np.testing.assert_equal(got_meta.left_flank_len,   ref_meta.left_flank_len)
        np.testing.assert_equal(got_meta.left_flank_start, ref_meta.left_flank_start)
        np.testing.assert_equal(got_meta.right_flank_len,   ref_meta.right_flank_len)
        np.testing.assert_equal(got_meta.right_flank_start, ref_meta.right_flank_start)
        if ref_meta.unannotated_segments_len.shape[0] > 0:
            np.testing.assert_equal(
                got_meta.unannotated_segments_len,
                ref_meta.unannotated_segments_len
            )
            np.testing.assert_equal(
                got_meta.unannotated_segments_start,
                ref_meta.unannotated_segments_start
            )

def test_viterbi(
    simple_data: SequenceDataset, simple_model: LearnMSAModel,
) -> None:
    """Test Viterbi algorithm and decoding."""
    length = [5, 3]

    # Use the model to predict Viterbi paths
    simple_model.viterbi_mode()
    simple_model.compile()
    viterbi_seqs = simple_model.predict(simple_data)
    viterbi_seqs = np.transpose(viterbi_seqs, (2, 0, 1))

    ref_seqs = np.array([
        # model 1 (FELIK - length 5)
        [[0, 1, 2, 3, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
         [9, 9, 9, 0, 1, 2, 3, 4, 12, 12, 12, 12, 12, 12, 12],
         [0, 1, 2, 3, 4, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12],
         [0, 1, 2, 3, 4, 10, 10, 10, 0, 1, 2, 3, 4, 11, 12],
         [9, 1, 2, 3, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
         [0, 1, 6, 6, 6, 2, 3, 4, 12, 12, 12, 12, 12, 12, 12],
         [0, 5, 5, 1, 2, 7, 3, 8, 8, 8, 4, 12, 12, 12, 12],
         [0, 1, 2, 7, 7, 7, 3, 4, 11, 11, 11, 12, 12, 12, 12]],
        # model 2 (AHC - length 3)
        [[5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
         [0, 1, 2, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
         [5, 5, 5, 5, 5, 5, 0, 2, 8, 8, 8, 8, 8, 8, 8],
         [5, 5, 5, 5, 5, 0, 1, 2, 6, 6, 6, 6, 6, 0, 8],
         [0, 3, 3, 3, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
         [5, 5, 0, 1, 2, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
         [5, 0, 1, 6, 6, 0, 6, 0, 1, 2, 7, 8, 8, 8, 8],
         [5, 5, 5, 0, 1, 2, 6, 6, 0, 1, 2, 8, 8, 8, 8]]
    ])

    np.testing.assert_equal(viterbi_seqs, ref_seqs)

    indices = np.array([
        [0, 3, 0, 0, 1, 0, 0, 0],
        [5, 0, 6, 5, 0, 2, 1, 3],
    ])  # skip the left flank

    # first domain hit
    ref_consensus = [  # model 1
        np.array([[0, 1, 2, 3, 4],
                    [3, 4, 5, 6, 7],
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [-1, 1, 2, 3, -1],
                    [0, 1, 5, 6, 7],
                    [0, 3, 4, 6, 10],
                    [0, 1, 2, 6, 7]]),
        # model 2
        np.array([[-1, -1, -1],
                    [0, 1, 2],
                    [6, -1, 7],
                    [5, 6, 7],
                    [0, 4, -1],
                    [2, 3, 4],
                    [1, 2, -1],
                    [3, 4, 5]])]
    ref_insertion_lens = [  # model1
        np.array([[0] * 4,
                    [0] * 4,
                    [0] * 4,
                    [0] * 4,
                    [0] * 4,
                    [0, 3, 0, 0],
                    [2, 0, 1, 3],
                    [0, 0, 3, 0]]),
        # model2
        np.array([[0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [3, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0]])]
    ref_insertion_start = [  # model1
        np.array([[-1] * 4,
                    [-1] * 4,
                    [-1] * 4,
                    [-1] * 4,
                    [-1] * 4,
                    [-1, 2, -1, -1],
                    [1, -1, 5, 7],
                    [-1, -1, 3, -1]]),
        # model2
        np.array([[-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1]])]
    ref_finished = np.array([  # model 1
        [True, True, True, False, True, True, True, True],
        # model 2
        [True, True, True, False, True, True, False, False]])
    ref_left_flank_lens = np.array([[0, 3, 0, 0, 1, 0, 0, 0],
                                    [5, 0, 6, 5, 0, 2, 1, 3]])
    ref_segment_lens = np.array([[0, 0, 0, 3, 0, 0, 0, 0],  # model 1
                                    [0, 0, 0, 5, 0, 0, 2, 2]])  # model 2
    ref_segment_start = np.array([[5, 8, 5, 5, 4, 8, 11, 8],  # model 1
                                    [5, 3, 8, 8, 5, 5, 3, 6]])  # model 2
    ref_right_flank_lens = np.array([[0, 0, 3, 1, 1, 0, 0, 3],  # model 1
                                        [0, 5, 0, 0, 0, 3, 1, 0]])  # model 2
    ref_right_flank_start = np.array([[5, 8, 5, 13, 4, 8, 11, 8],  # model 1
                                        [5, 3, 8, 14, 5, 5, 10, 11]])  # model 2

    s = len(simple_data.output_alphabet)
    A = SequenceDataset._default_alphabet.index("A")
    H = SequenceDataset._default_alphabet.index("H")
    C = SequenceDataset._default_alphabet.index("C")
    a = SequenceDataset._default_alphabet.index("A") + s
    h = SequenceDataset._default_alphabet.index("H") + s
    c = SequenceDataset._default_alphabet.index("C") + s
    F = SequenceDataset._default_alphabet.index("F")
    E = SequenceDataset._default_alphabet.index("E")
    L = SequenceDataset._default_alphabet.index("L")
    I = SequenceDataset._default_alphabet.index("I")
    X = SequenceDataset._default_alphabet.index("K")
    f = SequenceDataset._default_alphabet.index("F") + s
    e = SequenceDataset._default_alphabet.index("E") + s
    l = SequenceDataset._default_alphabet.index("L") + s
    i = SequenceDataset._default_alphabet.index("I") + s
    x = SequenceDataset._default_alphabet.index("K") + s
    GAP = s - 1
    gap = 2 * s - 1

    ref_left_flank_block = [np.array([[gap] * 3,  # model 1
                                        [a, h, c],
                                        [gap] * 3,
                                        [gap] * 3,
                                        [gap, gap, a],
                                        [gap] * 3,
                                        [gap] * 3,
                                        [gap] * 3]),
                            np.array([[gap, f, e, l, i, x],  # model 2
                                        [gap] * 6,
                                        [f, e, l, i, x, h],
                                        [gap, f, e, l, i, x],
                                        [gap] * 6,
                                        [gap, gap, gap, gap, f, e],
                                        [gap] * 5 + [f],
                                        [gap, gap, gap, f, e, l]])]
    ref_right_flank_block = [np.array([[gap] * 3,  # model 1
                                        [gap] * 3,
                                        [h, a, c],
                                        [a, gap, gap],
                                        [h, gap, gap],
                                        [gap] * 3,
                                        [gap] * 3,
                                        [a, h, c]]),
                                np.array([[gap] * 5,  # model 2
                                        [f, e, l, i, x],
                                        [gap] * 5,
                                        [gap] * 5,
                                        [gap] * 5,
                                        [l, i, x, gap, gap],
                                        [x] + [gap] * 4,
                                        [gap] * 5])]
    ref_ins_block = [np.array([[gap] * 2,
                                [gap] * 2,
                                [gap] * 2,
                                [gap] * 2,
                                [gap] * 2,
                                [gap] * 2,
                                [a, h],
                                [gap] * 2]),
                        np.array([[gap] * 3,
                                [gap] * 3,
                                [gap] * 3,
                                [gap] * 3,
                                [e, l, i],
                                [gap] * 3,
                                [gap] * 3,
                                [gap] * 3])]
    ref_core_blocks = [  # model 1
        [np.array([[F, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, X],
                    [F, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, X],
                    [F, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, X],
                    [F, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, X],
                    [GAP, gap, gap, E, gap, gap, gap, L, gap, gap, gap, I, gap, gap, gap, GAP],
                    [F, gap, gap, E, a, h, c, L, gap, gap, gap, I, gap, gap, gap, X],
                    [F, a, h, E, gap, gap, gap, L, a, gap, gap, I, a, h, c, X],
                    [F, gap, gap, E, gap, gap, gap, L, a, h, c, I, gap, gap, gap, X]]),
            np.array([[GAP] * 5,
                    [GAP] * 5,
                    [GAP] * 5,
                    [F, E, L, I, X],
                    [GAP] * 5,
                    [GAP] * 5,
                    [GAP] * 5,
                    [GAP] * 5])],
        # model 2
        [np.array([[GAP, gap, gap, gap, GAP, GAP],
                    [A, gap, gap, gap, H, C],
                    [A, gap, gap, gap, GAP, C],
                    [A, gap, gap, gap, H, C],
                    [A, e, l, i, H, GAP],
                    [A, gap, gap, gap, H, C],
                    [A, gap, gap, gap, H, GAP],
                    [A, gap, gap, gap, H, C]]),
            np.array([[GAP] * 3,
                    [GAP] * 3,
                    [GAP] * 3,
                    [A, GAP, GAP],
                    [GAP] * 3,
                    [GAP] * 3,
                    [A, GAP, GAP],
                    [A, H, C]])]]
    ref_num_blocks = [2, 3]
    # second domain hit
    ref_consensus_2 = [  # model 1
        np.array([[-1] * 5] * 3 +
                    [[8, 9, 10, 11, 12]] +
                    [[-1] * 5] * 4),
        # model 2
        np.array([[-1] * 3] * 3 +
                    [[13, -1, -1]] +
                    [[-1] * 3] * 2 +
                    [[5, -1, -1],
                    [8, 9, 10]])]
    ref_insertion_lens_2 = [np.array([[0] * 4] * 8),  # model 1
                            np.array([[0] * 2] * 8)]  # model 2
    ref_insertion_start_2 = [np.array([[-1] * 4] * 8),  # model 1
                                np.array([[-1] * 2] * 8)]  # model 2
    ref_finished_2 = np.array([
        [True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, False, True],
    ])
    ref_left_flank_lens_2 = np.array(
        [[0, 3, 0, 0, 1, 0, 0, 0],  # model 1
        [5, 0, 6, 5, 0, 2, 1, 3]]   # model 2
    )

    def assert_decoding_core_results(decoded, ref):
        for i in range(simple_data.num_seq):
            for d, r in zip(decoded, ref):
                np.testing.assert_equal(d[i], r[i])

    for i,L in enumerate(length):
        # test decoding
        # test first core block isolated
        decoding_core_results = decode_core_tf(
            L, viterbi_seqs[i], indices[i]
        )
        assert_decoding_core_results(
            decoding_core_results,
            (
                ref_consensus[i],
                ref_insertion_lens[i],
                ref_insertion_start[i],
                ref_finished[i]
            )
        )
        # test left flank insertions isolated
        left_flank_lens, left_flank_start = decode_flank_tf(
            viterbi_seqs[i],
            flank_state_id=2*L-1,
            indices=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        )
        np.testing.assert_equal(
            left_flank_lens, ref_left_flank_lens[i]
        )
        np.testing.assert_equal(
            left_flank_start, np.array([0, 0, 0, 0, 0, 0, 0, 0])
        )
        # test whole decoding
        meta_data = decode_tf(L, viterbi_seqs[i])
        assert meta_data.num_repeats == ref_num_blocks[i]
        all_rows_v = np.arange(simple_data.num_seq)
        dh0v, il0v, is0v, _, _ = meta_data.get_repeat_data(0, all_rows_v)
        dh1v, il1v, is1v, _, _ = meta_data.get_repeat_data(1, all_rows_v)
        assert_decoding_core_results(
            (dh0v, il0v, is0v, meta_data.skip[0]),
            (
                ref_consensus[i],
                ref_insertion_lens[i],
                ref_insertion_start[i],
                ref_finished[i]
            ),
        )
        assert_decoding_core_results(
            (dh1v, il1v, is1v, meta_data.skip[1]),
            (
                ref_consensus_2[i],
                ref_insertion_lens_2[i],
                ref_insertion_start_2[i],
                ref_finished_2[i],
            ),
        )
        np.testing.assert_equal(
            meta_data.left_flank_len_for(all_rows_v), ref_left_flank_lens[i]
        )
        np.testing.assert_equal(
            meta_data.left_flank_start_for(all_rows_v), np.array([0, 0, 0, 0, 0, 0, 0, 0])
        )
        uns_lv, uns_sv = meta_data.get_unannotated_data(0, all_rows_v)
        np.testing.assert_equal(uns_lv, ref_segment_lens[i])
        mask_v = ref_segment_lens[i] > 0
        np.testing.assert_equal(uns_sv[mask_v], ref_segment_start[i][mask_v])
        np.testing.assert_equal(
            meta_data.right_flank_len_for(all_rows_v), ref_right_flank_lens[i]
        )
        np.testing.assert_equal(
            meta_data.right_flank_start_for(all_rows_v), ref_right_flank_start[i]
        )

        # test conversion of decoded data to an actual alignment in table form
        # Prepare sequences array (integer render tokens) for all sequences
        output_len= len(simple_data.output_alphabet)
        sequences = np.zeros((simple_data.num_seq, simple_data.max_len), dtype=np.uint16)
        sequences += (output_len- 1)
        for j in range(simple_data.num_seq):
            l = simple_data.seq_lens[j]
            sequences[j, :l] = simple_data.get_encoded_seq(
                j, remap=False
            ).argmax(axis=-1)

        left_flank_block = AlignmentModel.get_insertion_block(
            sequences,
            meta_data.left_flank_len_for(all_rows_v),
            np.amax(meta_data.left_flank_len_for(all_rows_v)),
            meta_data.left_flank_start_for(all_rows_v),
            output_len,
            adjust_to_right=True,
        )
        np.testing.assert_equal(left_flank_block, ref_left_flank_block[i])
        right_flank_block = AlignmentModel.get_insertion_block(
            sequences,
            meta_data.right_flank_len_for(all_rows_v),
            np.amax(meta_data.right_flank_len_for(all_rows_v)),
            meta_data.right_flank_start_for(all_rows_v),
            output_len,
        )
        np.testing.assert_equal(right_flank_block, ref_right_flank_block[i])
        # just check the first insert for simplicity
        dh0_v, il0_v, is0_v, _, _ = meta_data.get_repeat_data(0, all_rows_v)
        ins_lens = il0_v[:, 0]
        ins_start = is0_v[:, 0]
        ins_block = AlignmentModel.get_insertion_block(
            sequences,
            ins_lens,
            np.amax(ins_lens),
            ins_start,
            output_len,
        )
        np.testing.assert_equal(ins_block, ref_ins_block[i])
        for ri, ref in enumerate(ref_core_blocks[i]):
            dh_riv, il_riv, is_riv, _, _ = meta_data.get_repeat_data(ri, all_rows_v)
            alignment_block = AlignmentModel.get_alignment_block(
                sequences,
                dh_riv,
                il_riv,
                np.amax(il_riv, axis=0),
                is_riv,
                output_len,
            )
            np.testing.assert_equal(alignment_block, ref)

