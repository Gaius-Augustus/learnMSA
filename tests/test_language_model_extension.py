import numpy as np
from learnMSA.legacy import Priors


def test_regularizer() -> None:
    # test the regularizer
    reg_shared = Priors.L2Regularizer(1, 1, True)
    reg_non_shared = Priors.L2Regularizer(1, 1, False)
    reg_shared.build([])
    reg_non_shared.build([])
    #just test the embedding part
    lengths = [5, 6]
    B = np.zeros((2, 20, 101), dtype=np.float32)
    B[0, :2*lengths[0]+2, 25:] = 2.
    B[0, 1:lengths[0]+1, 25:] = 3.
    B[1, :2*lengths[1]+2, 25:] = 5.
    B[1, 1:lengths[1]+1, 25:] = 4.
    r1 = reg_shared.get_l2_loss(B, lengths)
    r2 = reg_non_shared.get_l2_loss(B, lengths)
    assert all(r1[0,:-1] == 75 * 9 + 75 * 4)
    assert r1[0,-1] == 0
    assert all(r1[1,:-1] == 75 * 16 + 75 * 25)
    assert all(r2[0,:-1] == 75 * 9 + 7 * 75 * 4 / 5)
    assert r2[0,-1] == 0
    assert all(r2[1,:-1] == 75 * 16 + 8 * 75 * 25 / 6)
