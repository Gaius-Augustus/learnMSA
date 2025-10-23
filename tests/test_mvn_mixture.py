import numpy as np
from learnMSA.protein_language_models import MvnMixture
from learnMSA.msa_hmm import Utility


def test_mvn_single_diag_only() -> None:
    np.random.seed(77)
    mu = np.array([1., 2, 3, 4, 5], dtype=np.float32)
    d = mu.size
    scale_diag = np.array([.1, .5, 1, 2, 3], dtype=np.float32)
    scale = np.diag(scale_diag)
    inputs = np.random.multivariate_normal(mu, scale, size=100).astype(np.float32)
    # compute a reference assuming that tfp uses a correct implementation
    # precomputed to not have tfp as a dependency
    # import tensorflow_probability as tfp
    # ref_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=scale_diag)
    # ref_log_pdf = ref_dist.log_prob(inputs).numpy()
    ref_log_pdf = np.array([
        -4.9309063, -10.387703, -7.2909403, -4.436602, -27.024857,
        -5.253654, -8.531678, -10.450501, -7.292756, -9.677152,
        -5.506526, -12.509983, -46.371372, -4.3663635, -5.6152353,
        -7.4584303, -10.1104355, -8.395979, -4.0930266, -8.157899,
        -6.469728, -4.999018, -23.696964, -5.858695, -9.351412,
        -20.090115, -10.501094, -7.5617256, -11.118088, -6.100891,
        -8.797119, -8.742572, -7.511483, -19.419811, -16.507904,
        -14.794913, -6.568233, -9.369138, -8.830558, -6.8634834,
        -5.3114634, -8.607314, -10.14441, -13.3842535, -25.595154,
        -9.565649, -7.575345, -11.253255, -12.537072, -5.991749,
        -7.7170773, -6.0476046, -7.0488434, -5.262626, -6.590644,
        -11.621238, -7.5096426, -4.417821, -6.5862513, -5.6849046,
        -27.452187, -8.345725, -15.220712, -14.761422, -10.24355,
        -26.648506, -4.4250154, -7.253064, -9.48193, -9.329909,
        -10.536048, -20.788212, -7.585931, -9.671636, -16.005692,
        -7.0307503, -6.0479403, -9.651827, -9.062091, -5.6201506,
        -6.406321, -5.8290544, -5.5334306, -7.816556, -13.276981,
        -12.612909, -15.576953, -6.67165, -7.9874716, -11.401611,
        -12.275256, -4.94542, -8.66773, -10.191403, -4.8015766,
        -15.398995, -6.0817037, -13.620152, -4.935639, -5.3541765
    ])
    # reshape to match the expected shapes
    mu = np.reshape(mu, (1,1,1,d))
    scale_diag = np.reshape(scale_diag, (1,1,1,d))
    inputs = np.expand_dims(inputs, 0)
    # compute the log_prob using the custom implementation
    kernel = Utility.make_kernel(mu, scale_diag)
    dist = MvnMixture.MvnMixture(dim = d, kernel = kernel, diag_only = True)
    log_pdf = dist.log_pdf(inputs)
    # compare the results
    np.testing.assert_almost_equal(
        log_pdf[0,:,0].numpy(), ref_log_pdf, decimal=5
    )


def test_mvn_single_full() -> None:
    np.random.seed(1000)
    mu = np.array([1., 2, 3, 4, 5], dtype=np.float32)
    d = mu.size
    scale = np.random.rand(d,d).astype(np.float32)
    scale = np.matmul(scale, scale.T)
    #make triangular lower
    inputs = np.random.multivariate_normal(mu, scale, size=100).astype(np.float32)
    scale = np.tril(scale)
    # compute a reference assuming that tfp uses a correct implementation
    # precomputed to not have tfp as a dependency
    # import tensorflow_probability as tfp
    # ref_dist = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=scale)
    # ref_log_pdf = ref_dist.log_prob(inputs).numpy()
    ref_log_pdf = np.array([
        -7.7539816, -7.4303236, -7.687662, -7.848358, -8.034961,
        -7.8960657, -7.912344, -8.077688, -7.4278164, -7.7296476,
        -7.9770775, -7.7325077, -7.557103, -7.9264936, -7.5739584,
        -7.5619946, -7.9560127, -8.073469, -8.438643, -7.351403,
        -7.7048674, -7.4203863, -7.5072117, -7.4718685, -7.360958,
        -7.8579006, -7.5296745, -7.532502, -7.8499794, -7.718437,
        -7.3419724, -7.717458, -7.4757233, -7.524334, -7.77816,
        -7.513461, -7.831909, -7.32905, -7.4544945, -7.423876,
        -8.126367, -8.259831, -8.164715, -7.6520863, -7.459009,
        -7.765686, -7.4940863, -7.547223, -7.6232376, -7.4277477,
        -7.6923866, -7.448251, -7.502944, -7.5673, -7.422387,
        -7.4737186, -7.7795763, -7.917593, -7.46449, -7.79692,
        -7.6767826, -7.386978, -7.58654, -8.014507, -8.283238,
        -7.520664, -7.4706373, -7.746441, -7.4981394, -7.2982836,
        -7.358142, -7.3786783, -7.4688787, -8.411664, -8.404474,
        -7.556675, -9.168212, -8.249679, -7.396226, -7.7976904,
        -8.388175, -7.284189, -7.3497314, -7.3587933, -7.486101,
        -7.7001696, -7.9018383, -7.7343836, -8.025629, -7.430558,
        -7.668483, -7.534708, -7.740678, -7.649802, -7.9278097,
        -7.4801702, -7.7048597, -7.749136, -9.000921, -8.478548
    ])
    # reshape to match the expected shapes
    mu = np.reshape(mu, (1,1,1,d))
    scale = np.reshape(scale, (1,1,1,d,d))
    inputs = np.expand_dims(inputs, 0)
    # compute the log_prob using the custom implementation
    kernel = Utility.make_kernel(mu, scale)
    dist = MvnMixture.MvnMixture(dim = d, kernel = kernel, diag_only = False)
    log_pdf = dist.log_pdf(inputs)
    # compare the results
    np.testing.assert_almost_equal(
        log_pdf[0,:,0].numpy(), ref_log_pdf, decimal=2
    )
