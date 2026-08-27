"""PyTorch implementations of learnMSA's protein language model wrappers.

Importing this package pins the compute backend to PyTorch.
"""

from learnMSA.backend import set_backend

set_backend("pytorch")
