## Converts file format of priors and other saved models.
import DirichletMixture as dm

prior_path = "trained_prior/"

for n_comp in [1,3,9,32,64,128,256]:
    for dtype in ["float32", "float64"]:
        model_file = "_".join([str(n_comp), "True", dtype, "_dirichlet"])
        model = dm.load_mixture_model(prior_path + model_file, n_comp, 20, trainable=False, dtype=dtype)
        model.save_weights(prior_path + model_file + ".h5")

for n_comp in [1, 2, 3, 5, 10]:
    for dtype in ["float32", "float64"]:
        model_file = "_".join(["match_prior", str(n_comp),  dtype])
        model = dm.load_mixture_model(prior_path + "transition_priors/" + model_file, n_comp, 3, trainable=False, dtype=dtype)
        model.save_weights(prior_path + "transition_priors/" + model_file + ".h5")

for n_comp in [1, 2]:
    for dtype in ["float32", "float64"]:
        model_file = "_".join(["delete_prior", str(n_comp),  dtype])
        model = dm.load_mixture_model(prior_path + "transition_priors/" + model_file, n_comp, 2, trainable=False, dtype=dtype)
        model.save_weights(prior_path + "transition_priors/" + model_file + ".h5")

for n_comp in [1, 2, 3, 5]:
    for dtype in ["float32", "float64"]:
        model_file = "_".join(["insert_prior", str(n_comp),  dtype])
        model = dm.load_mixture_model(prior_path + "transition_priors/" + model_file, n_comp, 2, trainable=False, dtype=dtype)
        model.save_weights(prior_path + "transition_priors/" + model_file + ".h5")
