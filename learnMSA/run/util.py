import os
import json
import subprocess as sp
from argparse import ArgumentParser
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from learnMSA import Configuration

if TYPE_CHECKING:
    from learnMSA.util import EmbeddingDataset, SequenceDataset


DEFAULT_FALLBACK_MEMORY_MB = 4096


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
) -> None:
    """
    Args:
        cuda_visible_devices: str
            The value to set for the environment variable.
        verbose: bool
            Whether to enable std output messages.

    Sets up the GPU environment for TensorFlow based on the command line.
    Avoids importing TensorFlow until the user has set the CUDA_VISIBLE_DEVICES.
    This function should be called after parsing the command line arguments,
    otherwise just showing the help message will be slow.
    """
    if not cuda_visible_devices == "default":
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    # Must be set before any TensorFlow operations
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    import tensorflow as tf
    from tensorflow.python.client import device_lib

    # Check if multiple GPUs are installed / set in the environment variable
    if get_num_gpus() > 1:
        print(
            "Warning: Multiple GPUs detected. learnMSA currently does not "
            "support multi-GPU training. Only the first GPU will be used. "
            "However, you can distribute multiple learnMSA jobs to your GPUs "
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


def get_batch_multiplicator() -> int:
    """Returns the number of devices for batch size scaling.

    Returns the number of GPUs if at least one GPU is available,
    otherwise returns 1 (for CPU-only case).
    """
    return get_num_gpus() + int(get_num_gpus() == 0)


def get_gpu_memory() -> list[int]:
    """Returns a list of total memory (in MB) for each GPU detected."""
    if get_num_gpus() == 0:
        return []
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    try:
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    except (sp.CalledProcessError, FileNotFoundError, OSError) as e:
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

def get_avail_memory_bytes() -> float:
    """Returns the available VRAM in bytes. If no GPU is available, returns
    the available RAM in bytes."""
    gpu_memory = get_gpu_memory()
    if gpu_memory and gpu_memory[0] > 0:
        return float(gpu_memory[0]) * 1e6  # nvidia-smi reports MB

    # No usable GPU memory -> use currently available RAM
    try:
        # Linux/Unix: free physical pages * page size
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        ram_bytes = int(avail_pages) * int(page_size)
        if ram_bytes > 0:
            return float(ram_bytes)
    except (OSError, ValueError, AttributeError):
        pass

    return float(DEFAULT_FALLBACK_MEMORY_MB) * 1e6


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


def validate_output_file_requirements(config, parser) -> None:
    """Validate that output_file is provided when required.

    Output file is required unless:
    - Only --scores is used (and no other output options)
    - Only --save_model is used (and no other output options)
    - Only --logo is used (and no other output options)

    Args:
        config: Configuration object
        parser: Argument parser for error reporting
    """
    output_file_provided = config.input_output.output_file != Path()

    # Check if output_file is required
    if not output_file_provided:
        # Convert mode always requires output_file
        if config.input_output.convert:
            parser.error(
                "argument -o/--out_file is required when using --convert"
            )

        # If not using any output options, output_file is required for alignment
        using_scores = config.input_output.scores != Path()
        using_save_model = config.input_output.save_model != ""
        using_logo = config.visualization.logo != ""

        # If none of these alternative outputs are being used,
        # we need an output file
        if not (using_scores or using_save_model or using_logo):
            parser.error(
                "argument -o/--out_file is required (or use --scores, "
                "--save_model, or --logo to save alternative outputs)"
            )


def _flatten_nested_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested dictionaries by keeping leaf keys as destinations."""
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flattened.update(_flatten_nested_dict(value))
        else:
            flattened[key] = value
    return flattened


def _config_to_argparse_defaults(config: Configuration) -> dict[str, Any]:
    """Convert Configuration to argparse defaults with minimal special mapping."""
    conf = config.model_dump(mode="json")
    defaults = _flatten_nested_dict(conf)

    training = conf.get("training", {})
    input_output = conf.get("input_output", {})
    advanced = conf.get("advanced", {})

    if training.get("auto_crop", True):
        defaults["crop"] = "auto"
    elif training.get("crop", 0) == sys.maxsize:
        defaults["crop"] = "disable"
    else:
        defaults["crop"] = str(training.get("crop"))

    defaults["silent"] = not input_output.get("verbose", True)
    defaults["no_jit"] = not advanced.get("jit_compile", True)
    defaults["frozen_distances"] = not training.get("trainable_distances", True)

    return defaults


def apply_baseline_config_defaults(
    parser: ArgumentParser, config_path: str
) -> None:
    """Load baseline configuration from JSON and apply as parser defaults."""
    path = Path(config_path)
    try:
        with open(path, "r", encoding="utf-8") as file:
            config_dict = json.load(file)
    except OSError as exc:
        parser.error(f"Could not read config file '{path}': {exc}")
    except json.JSONDecodeError as exc:
        parser.error(f"Config file '{path}' is not valid JSON: {exc}")

    try:
        config = Configuration.model_validate(config_dict)
    except Exception as exc:
        parser.error(f"Invalid config file '{path}': {exc}")

    all_defaults = _config_to_argparse_defaults(config)
    parser_dests = {action.dest for action in parser._actions}
    defaults = {
        key: value for key, value in all_defaults.items() if key in parser_dests
    }
    parser.set_defaults(**defaults)

    for action in parser._actions:
        if not action.required:
            continue
        if action.dest not in defaults:
            continue
        value = defaults[action.dest]
        if value not in (None, "", []):
            action.required = False


def load_struct_data(
    config: Configuration, data: "SequenceDataset", stack: ExitStack
) -> "SequenceDataset | None":
    if config.input_output.struct_file is not None:
        from learnMSA.util import SequenceDataset
        struct_data = stack.enter_context(
            SequenceDataset(
                config.input_output.struct_file,
                "fasta",
                indexed=config.training.indexed_data,
                alphabet=config.structure.structural_alphabet,
                replace_with_x="",
                encode_as_one_hot=True,
            )
        )

        # Check if the data is valid
        struct_data.validate_dataset()
        if set(struct_data.seq_ids) != set(data.seq_ids):
            raise ValueError(
                "The sequence IDs in the structural dataset do not match "\
                "those in the input dataset."
            )
        struct_perm = [
            struct_data.seq_ids.index(seq_id) for seq_id in data.seq_ids
        ]
        struct_data.reorder(struct_perm)

        return struct_data
    return None

def load_emb_data(
    config: Configuration, data: "SequenceDataset", stack: ExitStack
) -> "EmbeddingDataset | None":
    if config.input_output.emb_file is not None:
        from learnMSA.util import EmbeddingDataset
        emb_data = stack.enter_context(
            EmbeddingDataset(config.input_output.emb_file)
        )
        # Reorder such that the embedding dataset has the same order of
        # sequences as the amino acid dataset
        emb_perm = [
            emb_data.seq_ids.index(seq_id) for seq_id in data.seq_ids
        ]
        emb_data.reorder(emb_perm)
        if config.input_output.verbose:
            print(
                f"Loaded embeddings from {config.input_output.emb_file} for " +
                f"{len(emb_data.seq_ids)} sequences"
            )
        return emb_data
    return None

