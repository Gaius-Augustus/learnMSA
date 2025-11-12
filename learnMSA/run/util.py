import os
import subprocess as sp
import warnings
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
        verbose : bool,
        grow_mem : bool = False
) -> None:
    """
    Args:
        cuda_visible_devices: str
            The value to set for the environment variable.
        verbose: bool
            Whether to enable std output messages.
        grow_mem: bool
            Whether to enable memory growth for GPUs.

    Sets up the GPU environment for TensorFlow based on the command line.
    Avoids importing TensorFlow until the user has set the CUDA_VISIBLE_DEVICES.
    This function should be called after parsing the command line arguments,
    otherwise just showing the help message will be slow.
    """
    if not cuda_visible_devices == "default":
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    # IMPORTANT: Memory growth must be set before any TensorFlow operations
    if grow_mem:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    import tensorflow as tf
    from tensorflow.python.client import device_lib

    # Check if multiple GPUs are installed / set in the environment variable
    if get_num_gpus() > 1:
        warnings.warn(
            "Multiple GPUs detected. learnMSA currently does not "\
            "support multi-GPU training. Only the first GPU will be used. "\
            "However, you can distribute multiple learnMSA jobs to your GPUs "\
            "by setting the parameter '-d'."
        )

    if verbose:
        GPUS = [
            x.physical_device_desc
            for x in device_lib.list_local_devices()
            if x.device_type == "GPU"
        ]
        if len(GPUS) == 0:
            if cuda_visible_devices == "-1" or \
                cuda_visible_devices == "":
                print(
                    "GPU disabled by user. Running on CPU instead. "\
                    "Expect slower performance especially for longer models."
                )
            else:
                print(
                    "It seems like no GPU is installed. Running on CPU "\
                    "instead. Expect slower performance especially for "\
                    "longer models."
                )
        else:
            print("Using GPU.")
        print("Found tensorflow version", tf.__version__)


def get_num_gpus() -> int:
    """Returns the number of GPUs detected by TensorFlow."""
    # import only if needed
    import tensorflow as tf
    num_gpu = len([x.name for x in tf.config.list_logical_devices() if x.device_type == 'GPU'])
    return num_gpu


def get_gpu_memory() -> list[int]:
    """Returns a list of total memory (in MB) for each GPU detected."""
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    try:
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    except sp.CalledProcessError as e:
        print(
            "Warning: There were GPU(s) detected, but nvidia-smi failed to "\
            "run. It is used to infer GPU memory and adapt the batch size. "\
            "Please make sure nvidia-smi is installed and working properly. "\
            "It might also mean that you are not running an NVIDIA GPU. "\
            "learnMSA will continue with default settings and might behave "\
            "as expected. You can adjust the batch size manually with the "\
            "-b option."
        )
        return []
    return memory_free_values


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
