# conftest.py
import os


def pytest_configure(config) -> None:
    """Set up framework-specific test environment for the selected backend."""
    backend = os.environ.get("LEARNMSA_BACKEND", "tensorflow")
    if backend == "tensorflow":
        # Without this TensorFlow grabs the whole GPU and later tests in the
        # session fail to allocate.
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    config.addinivalue_line(
        "markers", "tf: test exercises the TensorFlow backend"
    )
    config.addinivalue_line(
        "markers", "torch: test exercises the PyTorch backend"
    )
