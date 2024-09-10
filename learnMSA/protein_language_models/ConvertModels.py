## Converts file format of priors and other saved models.
import sys
sys.path.insert(0, "../..")
import Common as common
from BilinearSymmetric import make_scoring_model
from MvnPrior import make_pdf_model


prior_path = "priors_V3/"


def get_old_scoring_model_path(config : common.ScoringModelConfig):
    return f"new_scoring_models_frozen/{config.lm_name}_{config.dim}_{config.activation}{config.suffix}"

def get_old_prior_path(config : common.ScoringModelConfig, components):
    return f"priors_V3/{config.lm_name}_{config.dim}_reduced_mix{components}_{config.activation}{config.suffix}"


for plm in ["esm2", "proteinBERT", "protT5"]:
    for dim in [16,32,64,128]:
        for act in ["sigmoid", "softmax"]:
            # convert scoring model
            config = common.ScoringModelConfig(plm, dim, act)
            path = get_old_scoring_model_path(config)
            scoring_model = make_scoring_model(config, dropout=0.0, trainable=False) 
            scoring_model.load_weights(path + "/checkpoints")
            scoring_model.save_weights(path + ".h5")

            # convert priors
            for comp in [1, 10, 32, 100]:
                pdf_model = make_pdf_model(config, comp)
                prior_path = get_old_prior_path(config, comp)
                pdf_model.load_weights(prior_path + "/checkpoints")
                pdf_model.save_weights(prior_path + ".h5")
