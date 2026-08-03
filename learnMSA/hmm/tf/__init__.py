"""TensorFlow implementations of learnMSA's profile HMM layers.

Importing this package pins the compute backend to TensorFlow.
"""

from learnMSA.backend import set_backend

set_backend("tensorflow")
