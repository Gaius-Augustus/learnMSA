import sys
import os
sys.path.insert(0, os.path.dirname(__file__)+"/../../")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from learnMSA.protein_language_models.Argparse import make_scoring_model_argparser

parser = make_scoring_model_argparser()
parser.add_argument("--homfam_path", type=str, default="../../../../MSA-HMM-Analysis/data/homfam/refs/", help="The path to a folder containing the Homfam collection reference alignments.")

args = parser.parse_args()


if args.gpu != "default":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import json
import learnMSA.protein_language_models.Common as common 
import learnMSA.protein_language_models.TrainingUtil as training_util
import learnMSA.protein_language_models.DataPipeline as data
import tensorflow as tf


if __name__ == "__main__":
    language_model, encoder = common.get_language_model(args.lm, args.max_len, trainable=args.finetune)
    config = common.ScoringModelConfig(lm_name=args.lm, 
                                        dim=args.dim, 
                                        activation=args.activation,
                                        suffix=args.suffix,
                                        scaled=not args.unscaled)
    loss_func, acc_func, other_metrics = training_util.get_loss_and_metrics(config, 
                                                                            weight0=args.weight0, 
                                                                            weight1=args.weight1)
    full_model, scoring_model = training_util.make_full_scoring_model(encoder, 
                                                                      language_model, 
                                                                      config, 
                                                                      dropout=args.dropout, 
                                                                      trainable=False)
    checkpoint_path = os.path.dirname(__file__)+"/"+common.get_scoring_model_path(config)
    scoring_model.load_weights(checkpoint_path)
    print(tf.reduce_min(scoring_model.layers[-1].R), tf.reduce_max(scoring_model.layers[-1].R))
    training_util.compile_scoring_model(full_model, 
                                        loss_func, 
                                        acc_func, 
                                        categorical=args.activation=="softmax", 
                                        accumulate_gradients=args.acc, 
                                        other_metrics=other_metrics)
    test_ds, steps = data.make_homfam_dataset(encoder, args.batch, homfam_path=args.homfam_path)
    r = full_model.evaluate(test_ds, steps=steps-1, verbose=0)
    print(";".join([str(x) for x in r]))