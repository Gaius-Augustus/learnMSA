# conftest.py
import os

def pytest_configure() -> None:
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
