
import sys
import os
sys.path.insert(0, os.path.dirname(__file__)+"/../../")

from learnMSA.protein_language_models.Argparse import make_scoring_model_argparser

args = make_scoring_model_argparser().parse_args()

if args.gpu != "default":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import json
import learnMSA.protein_language_models.Common as common 
import learnMSA.protein_language_models.TrainingUtil as training_util
import tensorflow as tf


if __name__ == "__main__":
    print("Finetuning:", args.finetune)
    print(f"Maximum length is {args.max_len}. Will randomly crop sequences longer than this value.")
    print(tf.config.list_physical_devices('GPU'))
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
                                                                    trainable=True)
    checkpoint_path = os.path.dirname(__file__)+"/"+common.get_scoring_model_path(config)
    if args.resume:
        print("Resuming training from checkpoint ", checkpoint_path, "...")
        scoring_model.load_weights(checkpoint_path)
    else:
        print("Starting training...")
    if args.lr < 0:
        #compile with a learning rate schedule that decays over time
        lr = training_util.LinearWarmupSchedule(0.25, step_scale = 4/args.acc, warmup_steps=1000)
    else:
        lr = args.lr
    #custom callback tha only saves the prior, not the language model
    class PriorCheckpointCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if args.finetune:
                print("Finetuning currently not supported.")
                sys.exit()
                #scoring_model.save_weights(f"scoring_models/{args.lm}_{args.dim}_{args.activation}{args.suffix}/checkpoints")
                #language_model.model.save_weights(f"finetuned_models/{args.lm}_{args.dim}_{args.activation}{args.suffix}/checkpoints")
            else:
                scoring_model.save_weights(checkpoint_path)
            print(f"Saved checkpoint...")
    history = training_util.fit_scoring_model(encoder, 
                                             full_model, 
                                             loss_func,
                                             acc_func, 
                                             categorical=args.activation=="softmax",
                                             batch_size=args.batch, 
                                             accumulate_gradients=args.acc, 
                                             max_len=args.max_len, 
                                             lr=lr,
                                             callbacks=[PriorCheckpointCallback()], 
                                             other_metrics=other_metrics,
                                             epochs=10)
    json_path = checkpoint_path[:-len("/checkpoints")]+"/history.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    # save metadata and training history
    history.history["lm"] = args.lm
    history.history["lr"] = args.lr
    history.history["batch"] = args.batch
    history.history["max_protein_len"] = args.max_len
    history.history["dim"] = args.dim
    history.history["activation"] = args.activation
    history.history["suffix"] = args.suffix
    json.dump(history.history, open(json_path, 'w'))