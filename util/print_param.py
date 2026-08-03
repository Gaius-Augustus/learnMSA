import sys

import tensorflow as tf
import numpy as np

from learnMSA.model.tf import LearnMSAModel
from learnMSA.util import SequenceDataset
from learnMSA.model.batch_generator import BatchGenerator


def main():
    model_file = sys.argv[1]
    model = tf.keras.models.load_model(model_file)
    print("tau amino:\n", model.anc_probs_layer.make_tau()[:,:,0])
    print("tau 3Di:\n", model.anc_probs_layer.make_tau()[:,:,1])

if __name__ == "__main__":
    main()