"""PyTorch implementations of learnMSA's ancestral probability layers.

Importing this package pins the compute backend to PyTorch.
"""

from learnMSA.backend import set_backend

set_backend("pytorch")
