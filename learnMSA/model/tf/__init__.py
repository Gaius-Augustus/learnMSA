"""TensorFlow implementation of the learnMSA model.

Importing this package pins the compute backend to TensorFlow.
"""

from learnMSA.backend import set_backend

set_backend("tensorflow")

from .model import TFLearnMSAModel
