"""The torch wrappers must keep transformers from loading TensorFlow.

transformers < 5 imports TensorFlow whenever it is installed, which would put
both frameworks into one process. Neither development environment has both,
so the check is on the switch that prevents it rather than on ``sys.modules``.
"""

import os
import subprocess
import sys


def test_importing_the_torch_wrappers_disables_tensorflow() -> None:
    env = {k: v for k, v in os.environ.items() if k != "USE_TF"}
    probe = subprocess.run(
        [
            sys.executable, "-c",
            "import os\n"
            "import learnMSA.protein_language_models.torch\n"
            "print(os.environ.get('USE_TF'))",
        ],
        capture_output=True, text=True, env=env, check=True,
    )
    assert probe.stdout.strip() == "0"
