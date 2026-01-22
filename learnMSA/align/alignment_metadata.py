import numpy as np


# utility class used in AlignmentModel storing useful information on a
# specific alignment
class AlignmentMetaData():

    def __init__(
        self,
        core_blocks,
        left_flank,
        right_flank,
        unannotated_segments
    ):
        self.consensus = np.stack([C for C,_,_,_ in core_blocks])
        self.insertion_lens = np.stack([IL for _,IL,_,_ in core_blocks])
        self.insertion_start = np.stack([IS for _,_,IS,_ in core_blocks])
        self.finished = np.stack([f for _,_,_,f in core_blocks])
        self.left_flank_len = np.stack(left_flank[0])
        self.left_flank_start = np.stack(left_flank[1])
        self.right_flank_len = np.stack(right_flank[0])
        self.right_flank_start = np.stack(right_flank[1])
        if len(unannotated_segments) > 0:
            self.unannotated_segments_len = np.stack([
                l for l,_ in unannotated_segments
            ])
            self.unannotated_segments_start = np.stack([
                s for _,s in unannotated_segments
            ])
            self.unannotated_segment_lens_total = np.amax(
                self.unannotated_segments_len, axis=1
            )
        else:
            self.unannotated_segment_lens_total = 0
        self.num_repeats = self.consensus.shape[0]
        self.consensus_len = self.consensus.shape[-1]
        self.left_flank_len_total = np.amax(self.left_flank_len)
        self.right_flank_len_total = np.amax(self.right_flank_len)
        self.insertion_lens_total = np.amax(self.insertion_lens, axis=1)
        # convert at least 1 term to int32 in case of an alignment longer
        # than 32,767
        self.alignment_len = (
            self.left_flank_len_total.astype(np.int32) +
            self.consensus_len*self.num_repeats +
            np.sum(self.insertion_lens_total) +
            np.sum(self.unannotated_segment_lens_total) +
            self.right_flank_len_total
        )
