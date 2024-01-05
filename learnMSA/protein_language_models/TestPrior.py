import sys
import os
sys.path.insert(0, os.path.dirname(__file__)+"/../../")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from learnMSA.protein_language_models.Argparse import make_mvn_prior_argparser

parser = make_mvn_prior_argparser()
parser.add_argument("--homfam_path", type=str, default="../../../../MSA-HMM-Analysis/data/homfam/refs/", help="The path to a folder containing the Homfam collection reference alignments.")
args = parser.parse_args()

if args.gpu != "default":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import json
import tensorflow as tf
import learnMSA.protein_language_models.Common as common 
import learnMSA.protein_language_models.TrainingUtil as training_util
import learnMSA.protein_language_models.DataPipeline as data
from learnMSA.protein_language_models.MvnPrior import make_pdf_model, aggregate


if __name__ == "__main__":
    language_model, encoder = common.get_language_model(args.lm, args.max_len, trainable=False)
    emb_model = training_util.make_embedding_model(encoder, language_model)
    config = common.ScoringModelConfig(lm_name=args.lm, 
                                        dim=args.reduced_dim, 
                                        activation=args.activation, 
                                        suffix=args.suffix)
    log_pdf_model = make_pdf_model(config, num_components=args.components, aggregate_result=False)
    scoring_layer = training_util.get_scoring_layer(config)
    model = training_util.make_full_mvn_prior_model(emb_model, 
                                                    log_pdf_model, 
                                                    scoring_layer)
    # load weights
    checkpoint_path = os.path.dirname(__file__)+"/"+common.get_prior_path(config, args.components)
    log_pdf_model.load_weights(checkpoint_path)
    # compute log pdf of homfam embeddings
    model.compile(loss=lambda match_mask, log_pdf: aggregate(log_pdf, match_mask))
    test_ds, steps = data.make_homfam_dataset(encoder, 
                                                args.batch, 
                                                homfam_path=args.homfam_path, 
                                                for_prior=True)
    log_pdf_val = model.evaluate(test_ds, steps=steps-1, verbose=0)
    # compute log pdf of random embeddings
    reduced_emb_model = training_util.make_reduced_emb_model(emb_model, scoring_layer)
    # compute standard deviation of embeddings
    std = 0
    for batch in test_ds.take(steps-1):
        embeddings = reduced_emb_model(batch)
        std += tf.math.reduce_std(embeddings)
    std /= steps-1
    log_pdf_model.compile(loss=lambda match_mask, log_pdf: aggregate(log_pdf, match_mask))
    random_steps = 100
    random_test_ds = data.make_random_data(args.reduced_dim, args.batch, random_steps, scale=std)
    log_pdf_random_val = log_pdf_model.evaluate(random_test_ds, steps=random_steps, verbose=0)
    print(log_pdf_val, ";", log_pdf_random_val)