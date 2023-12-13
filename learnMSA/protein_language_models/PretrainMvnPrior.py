import sys
import os
sys.path.insert(0, os.path.dirname(__file__)+"/../../")

from learnMSA.protein_language_models.Argparse import make_mvn_prior_argparser

args = make_mvn_prior_argparser().parse_args()

if args.gpu != "default":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import json
import learnMSA.protein_language_models.Common as common 
import learnMSA.protein_language_models.TrainingUtil as training_util
from learnMSA.protein_language_models.MvnPrior import make_pdf_model



if __name__ == "__main__":
    print(f"Maximum length is {args.max_len}. Will randomly crop sequences longer than this value.")
    print("Loading language model...")
    language_model, encoder = common.get_language_model(args.lm, args.max_len, trainable=False)
    emb_model = training_util.make_embedding_model(encoder, language_model)
    config = common.ScoringModelConfig(lm_name=args.lm, 
                                        dim=args.reduced_dim, 
                                        activation=args.activation,
                                        suffix=args.suffix,
                                        scaled=not args.unscaled)
    log_pdf_model = make_pdf_model(config, 
                                    num_components=args.components,
                                    trainable=True,         
                                    aggregate_result=False)
    scoring_layer = training_util.get_scoring_layer(config) 
    checkpoint_path = os.path.dirname(__file__)+"/"+common.get_prior_path(config, args.components)
    if args.resume:
        print("Resuming training...")
        log_pdf_model.load_weights(checkpoint_path)
    else:
        print("Starting training...")
    history = training_util.fit_mvn_prior(encoder, 
                                        emb_model, 
                                        log_pdf_model, 
                                        scoring_layer, 
                                        args.batch, 
                                        args.max_len, 
                                        lr=args.lr, 
                                        epochs=args.epochs, 
                                        checkpoint_path=checkpoint_path)
    history.history["lm"] = args.lm
    history.history["lr"] = args.lr
    history.history["batch"] = args.batch
    history.history["max_protein_len"] = args.max_len
    history.history["reduced_dim"] = args.reduced_dim
    history.history["components"] = args.components
    history.history["activation"] = args.activation
    history.history["suffix"] = args.suffix
    json.dump(history.history, open(checkpoint_path[:-len("checkpoints/")]+"/history.json", 'w'))