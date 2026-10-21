"""PyTorch implementations of learnMSA's protein language model wrappers.

Importing this package pins the compute backend to PyTorch.
"""

import os

from learnMSA.backend import set_backend

set_backend("pytorch")

# transformers < 5 imports tensorflow when it is installed, and the backends
# must never be loaded into the same process. This runs before any encoder
# imports transformers, because the wrapper is resolved first.
os.environ.setdefault("USE_TF", "0")
