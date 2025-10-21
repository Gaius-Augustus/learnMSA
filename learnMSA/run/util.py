import os
from pathlib import Path


def get_version() -> str:
    """
    Get the version without importing learnMSA as a module.
    """
    base_dir = str(os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))
    version_file_path = base_dir + "/_version.py"
    with open(version_file_path, "rt") as version_file:
        version = version_file.readlines()[0].split("=")[1].strip(' "')
    return version


def setup_devices(
        cuda_visible_devices : str,
        silent : bool,
        grow_mem : bool = False
) -> None:
    """
    Args:
        cuda_visible_devices: str
            The value to set for the environment variable.
        silent: bool
            Whether to suppress output.
        grow_mem: bool
            Whether to enable memory growth for GPUs.

    Sets up the GPU environment for TensorFlow based on the command line.
    Avoids importing TensorFlow until the user has set the CUDA_VISIBLE_DEVICES.
    This function should be called after parsing the command line arguments,
    otherwise just showing the help message will be slow.
    """
    if not cuda_visible_devices == "default":
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    import tensorflow as tf

    # IMPORTANT: Memory growth must be set before any TensorFlow operations
    if grow_mem:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    from tensorflow.python.client import device_lib

    if not silent:
        GPUS = [
            x.physical_device_desc
            for x in device_lib.list_local_devices()
            if x.device_type == "GPU"
        ]
        if len(GPUS) == 0:
            if cuda_visible_devices == "-1" or \
                cuda_visible_devices == "":
                print(
                    "GPUs disabled by user. Running on CPU instead. "\
                    "Expect slower performance especially for longer models."
                )
            else:
                print(
                    "It seems like no GPU is installed. Running on CPU "\
                    "instead. Expect slower performance especially for "\
                    "longer models."
                )
        else:
            print("Using GPU(s):", GPUS)
        print("Found tensorflow version", tf.__version__)


def validate_filepath(filepath : str, expected_ext : str) -> Path:
    """Ensure filepath has the correct extension and return as Path object."""
    # Convert to Path object
    path = Path(filepath)

    # Get the extension
    ext = path.suffix

    # If no extension or wrong extension, append/replace with correct one
    if ext.lower() != expected_ext.lower():
        if not ext:  # No extension
            path = Path(str(path) + expected_ext)
        else:  # Wrong extension
            path = path.with_suffix(expected_ext)

    return path
