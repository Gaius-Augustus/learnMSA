from collections.abc import Sequence

import tensorflow as tf
from hidten.config import with_config
from hidten.tf import TFTransitioner

from learnMSA import TransitionerConfig, get_value


@with_config(TransitionerConfig)
class PHMMExplicitTransitioner(TFTransitioner):
    """A transitioner for explicit pHMMs with deletion states.
    This transitioner contains silent states and needs to be folded.

    The order of states in each head is:

    ``[left, begin, match_0, ..., match_L-1, insert_0, ..., insert_L-2, \
        delete_0, ..., delete_L-1, end, unannot, right, terminal]``

    where L is the number of match states in that head.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = TransitionerConfig(**kwargs)

        transitions, values = [], []
        for h, L in enumerate(self.config.lengths):
            # left flank
            transitions.append((h, 0, 0))
            transitions.append((h, 0, 1))
            values.append(get_value(self.config.p_left_left, h))
            values.append(1 - get_value(self.config.p_left_left, h))

            # begin to match 1 and delete 1
            transitions.append((h, 1, 2))
            values.append(get_value(self.config.p_begin_match, h, 0))
            # are transition probs. to the other match states provided?
            if isinstance(self.config.p_begin_match, Sequence)\
                    and isinstance(self.config.p_begin_match[h], Sequence):
                p_begin_match_inner = self.config.p_begin_match[h][1:]
                p_sum_prob_begin_match = sum(self.config.p_begin_match[h])
                assert p_sum_prob_begin_match <= 1, (
                    "Sum of p_begin_match for head {h} is "
                    f"{p_sum_prob_begin_match}, which is > 1"
                )
                p_begin_delete = 1 - p_sum_prob_begin_match
            else:
                p = get_value(self.config.p_begin_match, h, 0)
                p_begin_match_inner = (1-p) / (L-1)
                p_begin_delete = get_value(self.config.p_begin_delete, h)
            transitions.append((h, 1, 2*L+1))
            values.append(p_begin_delete)

            for i in range(L - 1):
                # begin to match i+1
                transitions.append((h, 1, i+3))
                values.append(
                    p_begin_match_inner[i]
                    if isinstance(p_begin_match_inner, Sequence)
                    else p_begin_match_inner
                )

                # match to match
                transitions.append((h, i+2, i+3))
                values.append(get_value(self.config.p_match_match, h, i))

                # match to insert
                transitions.append((h, i+2, L+i+2))
                values.append(get_value(self.config.p_match_insert, h, i))

                # self-loop in insert
                transitions.append((h, L+i+2, L+i+2))
                values.append(get_value(self.config.p_insert_insert, h, i))

                # insert to match
                transitions.append((h, L+i+2, i+3))
                values.append(1 - get_value(self.config.p_insert_insert, h, i))

                # match to delete
                transitions.append((h, i+2, 2*L+i+2))
                values.append(
                    1 - get_value(self.config.p_match_match, h, i)
                    - get_value(self.config.p_match_insert, h, i)
                    - get_value(self.config.p_match_end, h, i)
                )

                # delete to delete
                transitions.append((h, 2*L+i+1, 2*L+i+2))
                values.append(get_value(self.config.p_delete_delete, h, i))

            # match L to end
            transitions.append((h, L+2, 3*L+1))
            values.append(1.0)

            # delete L to end
            transitions.append((h, 3*L, 3*L+1))
            values.append(1.0)

            # end to unannot
            transitions.append((h, 3*L+1, 3*L+2))
            values.append(get_value(self.config.p_end_unannot, h))

            # unannot to unannot
            transitions.append((h, 3*L+2, 3*L+2))
            values.append(get_value(self.config.p_unannot_unannot, h))

            # unannot to begin
            transitions.append((h, 3*L+2, 1))
            values.append(1 - get_value(self.config.p_unannot_unannot, h))

            # end to right
            transitions.append((h, 3*L+1, 3*L+3))
            values.append(get_value(self.config.p_end_right, h))

            # right to right
            transitions.append((h, 3*L+3, 3*L+3))
            values.append(get_value(self.config.p_right_right, h))

            # right to terminal
            transitions.append((h, 3*L+3, 3*L+4))
            values.append(1 - get_value(self.config.p_right_right, h))

            # terminal to terminal
            transitions.append((h, 3*L+4, 3*L+4))
            values.append(1.0)

        # starting distribution
        start, start_values = [], []
        for h in range(len(self.config.lengths)):
            start.extend([(h, 0), (h, 1)])
            start_values.extend([
                get_value(self.config.p_start_left_flank, h),
                1 - get_value(self.config.p_start_left_flank, h)
            ])

        self.allow = transitions
        self.initializer = values
        self.allow_start = start
        self.initializer_start = start_values