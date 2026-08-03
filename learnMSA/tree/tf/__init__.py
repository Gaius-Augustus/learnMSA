"""TensorFlow implementations of learnMSA's evolutionary tree layers.

Importing this package pins the compute backend to TensorFlow, which also
initializes evoten's backend facade that the substitution model math runs on.
"""

from learnMSA.backend import set_backend

set_backend("tensorflow")
