import numpy as np

from learnMSA.msa_hmm import AncProbsLayer, DirichletMixture, Initializers


def test_dirichlet_log_pdf_single() -> None:
    epsilon = 1e-16
    alphas = np.array([[1., 1., 1.], [1., 2, 3], [50., 50., 50.], [100., 1., 10.]])
    probs = np.array([[.2, .3, .5], [1.-2*epsilon, epsilon, epsilon], [.8, .1, .1], [.3, .3, .4]])
    expected = np.array([[0.693146, 0.693146, 0.693146, 0.693146],
                        [1.5040779, -106.42974, -2.8134103, 1.0577908],
                        [-5.509186, -3444.141, -70.27524, 3.4245605],
                        [-127.1859, -293.1855, -4.427696, -89.05315]])
    q = np.array([1.])
    for e, alpha in zip(expected, alphas):
        alpha = np.expand_dims(alpha, 0)
        log_pdf = DirichletMixture.dirichlet_log_pdf(probs, alpha, q)
        np.testing.assert_almost_equal(log_pdf, e, decimal=3)
        alpha_init = Initializers.ConstantInitializer(
            AncProbsLayer.inverse_softplus(alpha).numpy()
        )
        mix_init = Initializers.ConstantInitializer(np.log(q))
        mean_log_pdf = DirichletMixture.DirichletMixtureLayer(
            1, 3, alpha_init=alpha_init, mix_init=mix_init
        )(probs)
        np.testing.assert_almost_equal(mean_log_pdf, np.mean(e), decimal=3)


def test_dirichlet_log_pdf_mix() -> None:
    epsilon = 1e-16
    alpha = np.array([[1., 1., 1.], [1., 2, 3], [50., 50., 50.], [100., 1., 10.]])
    probs = np.array([[.2, .3, .5], [1.-2*epsilon, epsilon, epsilon], [.8, .1, .1], [.3, .3, .4]])

    expected = np.array([0.48613059, -0.69314836, -0.65780917, 2.1857463])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    log_pdf = DirichletMixture.dirichlet_log_pdf(probs, alpha, q)
    np.testing.assert_almost_equal(log_pdf, expected, decimal=3)
    alpha_init = Initializers.ConstantInitializer(
        AncProbsLayer.inverse_softplus(alpha).numpy()
    )
    mix_init = Initializers.ConstantInitializer(np.log(q))
    mean_log_pdf = DirichletMixture.DirichletMixtureLayer(
        4, 3, alpha_init=alpha_init, mix_init=mix_init
    )(probs)
    np.testing.assert_almost_equal(mean_log_pdf, np.mean(expected), decimal=3)

    expected2 = np.array([0.39899244, 0.33647106, 0.33903092, 1.36464418])
    q2 = np.array([0.7, 0.02, 0.08, 0.2])
    log_pdf2 = DirichletMixture.dirichlet_log_pdf(probs, alpha, q2)
    np.testing.assert_almost_equal(log_pdf2, expected2, decimal=3)
    mix_init2 = Initializers.ConstantInitializer(np.log(q2))
    mean_log_pdf2 = DirichletMixture.DirichletMixtureLayer(
        4, 3, alpha_init=alpha_init, mix_init=mix_init2
    )(probs)
    np.testing.assert_almost_equal(mean_log_pdf2, np.mean(expected2), decimal=3)
