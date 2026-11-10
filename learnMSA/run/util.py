import os
import json
import subprocess as sp
import sys
from argparse import ArgumentParser
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import learnMSA.backend as backend
from learnMSA import Configuration
from learnMSA.backend import get_backend

if TYPE_CHECKING:
    from learnMSA.util import (EmbeddingCache, EmbeddingDataset,
                               SequenceDataset)
    from learnMSA.util.redundancy import RedundancyMap


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
    one_dnn_opts: bool,
) -> None:
    """
    Args:
        cuda_visible_devices: str
            The value to set for the environment variable.
        verbose: bool
            Whether to enable std output messages.

    Sets up the GPU environment of the selected backend based on the command
    line. Avoids importing the framework until the user has set the
    CUDA_VISIBLE_DEVICES. This function should be called after parsing the
    command line arguments, otherwise just showing the help message will be
    slow.
    """
    # Honoured by every backend.
    if not cuda_visible_devices == "default":
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    if get_backend() == "tensorflow":
        _setup_tensorflow_env(one_dnn_opts)
    backend.configure_runtime()

    # Check if multiple GPUs are installed / set in the environment variable
    if get_num_gpus() > 1:
        print(
            "Warning: Multiple GPUs detected. learnMSA currently does not "
            "support multi-GPU training. Only the first GPU will be used. "
            "However, you can distribute multiple learnMSA jobs to your GPUs "
            "by setting the parameter '-d'."
        )

    if verbose:
        if get_num_gpus() == 0:
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
        print(f"Found {get_backend()} version", backend.framework_version())


def _setup_tensorflow_env(one_dnn_opts: bool) -> None:
    """Environment TensorFlow needs set before it is first imported."""
    # triton segfaults when TensorFlow's CUDA libs load in the same process.
    # Only poison the module when nothing has imported it yet; the torch
    # backend needs the real thing (hidten[pytorch] requires triton).
    if "triton" not in sys.modules:
        sys.modules["triton"] = None  # type: ignore[assignment]

    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    if not one_dnn_opts:
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Hide TensorFlow/absl startup logs. Defaults, so users can still override
    # them through the environment.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")


def get_num_gpus() -> int:
    """Returns the number of GPUs detected by the selected backend."""
    return backend.num_gpus()


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
        memory_total_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_total_values = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]
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
    return memory_total_values

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


def resolve_input_file(config, parser) -> None:
    """Resolve the input file when -i is omitted.

    When no input file is given, either --struct is a directory of
    structures, whose amino acid sequences are inferred with Foldseek, or
    both --from_msa and --save_model must be provided so that the MSA can be
    converted directly into a saved model. In that case the MSA file is used
    as the sequence input.

    When -i is given together with a directory of structures, see
    ``select_structure_sequences``.

    Args:
        config: Configuration object (mutated in-place when resolved)
        parser: Argument parser for error reporting
    """
    struct_file = config.input_output.struct_file
    from_structures = struct_file is not None and Path(struct_file).is_dir()
    if config.input_output.input_file != Path():
        if from_structures:
            select_structure_sequences(config, parser)
        return

    if from_structures:
        infer_sequences_from_structures(config, parser)
        return

    if config.init_msa.from_msa is None or config.input_output.save_model == "":
        parser.error(
            "argument -i/--in_file is required unless --struct is a "
            "directory of structures or both --from_msa and --save_model "
            "are provided"
        )
    config.input_output.input_file = config.init_msa.from_msa


def infer_sequences_from_structures(config, parser) -> None:
    """Computes the amino acid sequences and the 3Di letters of the
    structures in the --struct directory with Foldseek, writes both as FASTA
    to the work dir and uses them as -i and --struct.

    Fails with a request for -i if a sequence cannot be inferred, i.e. it is
    missing or has a different length than its 3Di. Unknown residues are
    kept as X, which learnMSA scores with the background distribution, with
    a warning.

    Args:
        config: Configuration object (mutated in-place)
        parser: Argument parser for error reporting
    """
    from learnMSA.structure.foldseek import structures_to_3di

    pdb_dir = Path(config.input_output.struct_file)
    work_dir = Path(config.input_output.work_dir)
    stem = pdb_dir.resolve().name
    aa_file = work_dir / f"{stem}.fasta"
    struct_file = work_dir / f"{stem}.3di.fasta"
    structures_to_3di(
        pdb_dir, struct_file, work_dir, aa_output=aa_file,
        verbose=config.input_output.verbose,
    )

    aa_seqs, struct_seqs = _read_fasta(aa_file), _read_fasta(struct_file)
    if not aa_seqs:
        parser.error(f"Foldseek found no structures in {pdb_dir}.")
    failed = [
        seq_id for seq_id in struct_seqs.keys() | aa_seqs.keys()
        if len(aa_seqs.get(seq_id, "")) != len(struct_seqs.get(seq_id, ""))
    ]
    if failed:
        parser.error(
            f"Could not infer the amino acid sequence of {len(failed)} "
            f"structure(s) in {pdb_dir}, e.g. {sorted(failed)[0]} (missing "
            "or a different length than its 3Di sequence). Please "
            "provide the sequences with -i/--in_file; their IDs must equal "
            "the Foldseek IDs (file name without extension, plus _<chain> "
            "for files with several chains)."
        )
    unknown = sorted(
        seq_id for seq_id, seq in aa_seqs.items() if "X" in seq.upper()
    )
    if config.input_output.verbose:
        print(
            f"Inferred {len(aa_seqs)} amino acid sequences from the "
            f"structures in {pdb_dir}: {aa_file}"
        )
        if unknown:
            print(
                f"Warning: {len(unknown)} structure(s) have unknown residues "
                f"(e.g. {unknown[0]}), which are aligned as X. Provide the "
                "sequences with -i/--in_file if they are known."
            )
    config.input_output.input_file = aa_file
    config.input_output.struct_file = struct_file


def _has_structure_files(pdb_dir: Path) -> bool:
    return any(path.is_file() for path in pdb_dir.rglob("*"))


def _read_fasta(path: Path) -> dict[str, str]:
    from Bio import SeqIO

    return {r.id: str(r.seq) for r in SeqIO.parse(path, "fasta")}


def _standardize(seq: str) -> str:
    return seq.upper().replace("-", "").replace(".", "")


def select_structure_sequences(config, parser) -> None:
    """Prepares -i together with a --struct directory that may hold
    structures for only some of the input sequences. The set of IDs from the
    structures must be a subset of the IDs in -i. Structures whose ID is not
    in -i are an error. Sequences that are in both inputs should agree; if they
    don't, the structure file's sequence takes precedence, with a warning.

    Args:
        config: Configuration object (mutated in-place)
        parser: Argument parser for error reporting
    """
    from Bio import SeqIO

    from learnMSA.structure.foldseek import structures_to_3di

    io = config.input_output
    verbose = io.verbose
    pdb_dir = Path(io.struct_file)
    input_file = Path(io.input_file)
    work_dir = Path(io.work_dir)
    if not _has_structure_files(pdb_dir):
        if verbose:
            print(
                f"Warning: {pdb_dir} contains no structure files. The 3Di "
                "letters of all sequences are predicted with ProstT5."
            )
        io.struct_file = None
        config.structure.predict_3di = True
        return

    work_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem
    struct_file = work_dir / f"{stem}.3di.fasta"
    aa_file = work_dir / f"{stem}.structure_aa.fasta"
    structures_to_3di(
        pdb_dir, struct_file, work_dir, aa_output=aa_file, verbose=verbose,
    )
    struct_seqs, aa_seqs = _read_fasta(struct_file), _read_fasta(aa_file)

    # Only the input sequences that have a structure are kept in memory.
    input_seqs: dict[str, str] = {}
    num_input = 0
    for record in SeqIO.parse(input_file, io.input_format):
        num_input += 1
        if record.id in struct_seqs:
            input_seqs[record.id] = str(record.seq)
    extra = sorted(struct_seqs.keys() - input_seqs.keys())
    if extra:
        parser.error(
            f"{len(extra)} structure(s) in {pdb_dir} have no sequence in "
            f"{input_file}, e.g. {extra[0]}. Foldseek names a structure "
            "after its file without the extension, plus _<chain> for files "
            "with several chains."
        )

    replaced: dict[str, str] = {}
    rescued: list[str] = []
    ignored: set[str] = set()
    for seq_id, letters in struct_seqs.items():
        aa = aa_seqs.get(seq_id, "")
        seq = _standardize(input_seqs[seq_id])
        if len(aa) == len(letters):
            if aa.upper() != seq:
                replaced[seq_id] = aa
        elif len(seq) == len(letters):
            rescued.append(seq_id)
        else:
            ignored.add(seq_id)
    num_struct = len(struct_seqs) - len(ignored)

    if verbose:
        others = (
            f", the others with the sequence in {input_file}"
            if num_struct < num_input else ""
        )
        print(
            f"{num_struct} of {num_input} sequences have a structure in "
            f"{pdb_dir}. They are aligned with the amino acid sequence of "
            f"their structure file{others}."
        )
    if replaced:
        new_input = work_dir / f"{stem}.input.fasta"
        with open(new_input, "w") as f:
            for record in SeqIO.parse(input_file, io.input_format):
                seq = replaced.get(record.id, str(record.seq))
                f.write(f">{record.id}\n{seq}\n")
        if verbose:
            example = sorted(replaced)[0]
            print(
                f"Warning: {len(replaced)} sequence(s) in {input_file} "
                f"differ from the sequence of their structure file (e.g. "
                f"{example}). learnMSA aligns the structure sequences, "
                f"see {new_input}."
            )
        io.input_file = new_input
        io.input_format = "fasta"
    if rescued and verbose:
        print(
            f"Warning: the amino acid sequence of {len(rescued)} "
            f"structure(s) could not be inferred (e.g. {sorted(rescued)[0]}). "
            f"Their sequence in {input_file} is used instead, since it has "
            "the length of the 3Di sequence."
        )
    if ignored:
        if verbose:
            print(
                f"Warning: {len(ignored)} structure(s) are ignored (e.g. "
                f"{sorted(ignored)[0]}): neither the structure file nor "
                f"{input_file} has an amino acid sequence of the length of "
                "their 3Di sequence. Their 3Di is predicted like that of "
                "sequences without a structure."
            )
        with open(struct_file, "w") as f:
            for seq_id, letters in struct_seqs.items():
                if seq_id not in ignored:
                    f.write(f">{seq_id}\n{letters}\n")
    if num_struct == 0:
        io.struct_file = None
        config.structure.predict_3di = True
    else:
        io.struct_file = struct_file


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

    if config.input_output.posterior_full \
            and config.input_output.posterior_file == Path():
        parser.error("--posterior_full requires --posterior_file")

    # Check if output_file is required
    if not output_file_provided:
        # Convert mode always requires output_file
        if config.input_output.convert:
            parser.error(
                "argument -o/--out_file is required when using --convert"
            )

        # If not using any output options, output_file is required for alignment
        using_scores = config.input_output.scores != Path()
        using_decode = config.input_output.decode_file != Path() \
            or config.input_output.posterior_file != Path()
        using_save_model = config.input_output.save_model != ""
        using_logo = config.visualization.plot != ""

        # If none of these alternative outputs are being used,
        # we need an output file
        if not (using_scores or using_decode or using_save_model
                or using_logo):
            parser.error(
                "argument -o/--out_file is required (or use --scores, "
                "--decode_file, --posterior_file, --save_model, or --logo to "
                "save alternative outputs)"
            )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def merge_config_file(
    config_path: str, base_config: Configuration, parser: ArgumentParser
) -> Configuration:
    """Load a JSON config file and merge it on top of base_config.

    Values from the file override those already in base_config, so --config
    takes the highest priority in the resolution chain:
      program defaults → CLI args → --config file

    Args:
        config_path: Path to the JSON configuration file.
        base_config: Configuration object to merge into.
        parser: Argument parser used for error reporting.

    Returns:
        A new Configuration with the file values applied on top.
    """
    path = Path(config_path)
    try:
        with open(path, "r", encoding="utf-8") as file:
            config_dict = json.load(file)
    except OSError as exc:
        parser.error(f"Could not read config file '{path}': {exc}")
    except json.JSONDecodeError as exc:
        parser.error(f"Config file '{path}' is not valid JSON: {exc}")

    merged = _deep_merge(base_config.model_dump(mode="json"), config_dict)
    try:
        return Configuration.model_validate(merged)
    except Exception as exc:
        parser.error(f"Invalid config file '{path}': {exc}")


def load_struct_data(
    config: Configuration,
    data: "SequenceDataset",
    stack: ExitStack | None = None,
    reuse: "RedundancyMap | None" = None,
) -> "SequenceDataset | EmbeddingDataset | None":
    """Loads the structural track: a 3Di FASTA file, an ``.npz`` of
    per-residue 3Di logits (e.g. from ``learnMSA-3di``), a directory of
    PDB/mmCIF files (3Di computed with Foldseek into the work dir) or, with
    ``structure.predict_3di``, 3Di logits predicted in memory (only for the
    representatives in ``reuse``, if given).

    A file or directory may cover only some of the sequences in ``data``.
    The 3Di of the others is then predicted with ProstT5, see
    ``_complete_struct_logits``."""
    from learnMSA.structure.io import is_logits_file

    struct_file = config.input_output.struct_file
    from_structures = (
        not config.structure.predict_3di
        and struct_file is not None
        and Path(struct_file).is_dir()
    )
    if config.structure.predict_3di:
        struct_data = _predict_struct_logits(config, data, reuse)
    elif struct_file is None:
        return None
    elif is_logits_file(struct_file):
        struct_data = _load_struct_logits(config)
    else:
        if from_structures:
            from learnMSA.structure.foldseek import structures_to_3di

            work_dir = Path(config.input_output.work_dir)
            struct_file = work_dir / (
                Path(config.input_output.input_file).stem + ".3di.fasta"
            )
            structures_to_3di(
                config.input_output.struct_file, struct_file, work_dir,
                verbose=config.input_output.verbose,
            )
        config.structure.input_format = "tokens"
        from learnMSA.util import SequenceDataset
        dataset = SequenceDataset(
            struct_file,
            "fasta",
            indexed=config.training.indexed_data,
            alphabet=config.structure.structural_alphabet,
            remap=False,
            unknown_ends=config.structure.mask_ends,
        )
        if stack is None:
            struct_data = dataset
        else:
            struct_data = stack.enter_context(dataset)

        # Check if the data is valid. A single structure is fine when it
        # covers only some of the sequences.
        struct_data.validate_dataset(single_seq_ok=True)
    index = {seq_id: i for i, seq_id in enumerate(struct_data.seq_ids)}
    data_ids = set(data.seq_ids)
    extra = [seq_id for seq_id in struct_data.seq_ids
             if seq_id not in data_ids]
    if extra:
        hint = (
            " Foldseek names a structure after its file without the "\
            "extension, plus _<chain> for files with several chains."
            if from_structures else ""
        )
        raise ValueError(
            "The sequence IDs in the structural dataset do not match "\
            f"those in the input dataset: {len(extra)} are not in the "\
            f"input, e.g. {extra[0]}." + hint
        )
    mismatch = [
        (seq_id, int(struct_data.seq_lens[index[seq_id]]), int(length))
        for seq_id, length in zip(data.seq_ids, data.seq_lens)
        if seq_id in index and struct_data.seq_lens[index[seq_id]] != length
    ]
    if mismatch:
        seq_id, struct_len, length = mismatch[0]
        raise ValueError(
            f"{len(mismatch)} sequences differ in length between the "
            f"structural dataset and the input, e.g. {seq_id}: "
            f"{struct_len} vs {length}."
        )
    if len(index) < data.num_seq:
        return _complete_struct_logits(config, data, struct_data)
    struct_data.adapt_order(data)
    return struct_data


def _complete_struct_logits(
    config: Configuration,
    data: "SequenceDataset",
    struct_data: "SequenceDataset | EmbeddingDataset",
) -> "EmbeddingDataset":
    """The structural track of all sequences in ``data`` when
    ``struct_data`` covers only some of them.

    The 3Di of sequences with missing structures is predicted with ProstT5.
    Everything is stored as logits: 3Di letters are encoded such that the
    observation layer maps them back to their one-hot vectors.
    """
    from learnMSA.structure.io import tokens_as_logits
    from learnMSA.util import EmbeddingCache, EmbeddingDataset

    index = {seq_id: i for i, seq_id in enumerate(struct_data.seq_ids)}
    if config.language_model.plm_cluster_id < 1:
        raise ValueError(
            f"--struct covers only {len(index)} of {data.num_seq} sequences. "
            "--plm_cluster_id can not be used with such mixed structural "
            "inputs, only with --struct without a path (ProstT5 predicts the "
            "3Di of all sequences). Please remove --plm_cluster_id."
        )
    _require_torch_for_logits()
    st = config.structure
    from_tokens = st.input_format == "tokens"
    missing = data.subset(np.array(
        [i for i, seq_id in enumerate(data.seq_ids) if seq_id not in index]
    ))
    if config.input_output.verbose:
        print(
            f"{len(index)} of {data.num_seq} sequences have a structure. "
            f"The 3Di of the other {missing.num_seq} is predicted with "
            "ProstT5."
        )
    predicted = _predict_struct_logits(config, missing)

    seq_lens = np.asarray(data.seq_lens, dtype=np.int64)
    starts = np.cumsum(seq_lens) - seq_lens
    dim = len(st.structural_alphabet)
    cache = np.empty((int(seq_lens.sum()), dim), dtype=np.float16)
    k = 0
    for i, seq_id in enumerate(data.seq_ids):
        if seq_id in index:
            z = struct_data.get_encoded_seq(index[seq_id])
            if from_tokens:
                z = tokens_as_logits(
                    z, st.background_distribution, st.input_temperature
                )
        else:
            z = predicted.get_encoded_seq(k)
            k += 1
        cache[starts[i]:starts[i] + seq_lens[i]] = z
    return EmbeddingDataset(
        embedding_cache=EmbeddingCache(seq_lens, dim, cache=cache),
        seq_ids=list(data.seq_ids),
    )


def _require_torch_for_logits() -> None:
    # Checked before anything imports torch: a process must never load both
    # frameworks.
    if get_backend() != "pytorch":
        raise NotImplementedError(
            "Per-residue 3Di logits (an .npz file, --struct without a path "
            "or structures for only some of the sequences) are only "
            "supported by the PyTorch backend. Use --backend pytorch or a "
            "3Di FASTA file for all sequences."
        )


def _match_alphabet(
    dataset: "EmbeddingDataset", stored: str, alphabet: str, source: str
) -> None:
    """Permutes the logit columns of ``dataset`` from the letter order
    ``stored`` to ``alphabet``; rejects a different set of letters."""
    if sorted(stored) != sorted(alphabet) or len(set(stored)) != len(stored):
        raise ValueError(
            f"The 3Di letters of {source} ({stored}) do not match the "
            f"structural alphabet ({alphabet})."
        )
    if stored != alphabet:
        dataset.reorder_dims([stored.index(c) for c in alphabet])


def _load_struct_logits(config: Configuration) -> "EmbeddingDataset":
    """Per-residue 3Di logits; marks the structural input as logits.

    The columns are permuted if the file stores the letters of the
    structural alphabet in another order. A file over a different set of
    letters is rejected.
    """
    from learnMSA.util import EmbeddingDataset

    _require_torch_for_logits()
    config.structure.input_format = "logits"
    filepath = config.input_output.struct_file
    dataset = EmbeddingDataset(filepath)
    _match_alphabet(dataset, str(dataset.metadata["alphabet"]),
                    config.structure.structural_alphabet, str(filepath))
    return dataset


def _predict_struct_logits(
    config: Configuration,
    data: "SequenceDataset",
    reuse: "RedundancyMap | None" = None,
) -> "EmbeddingDataset":
    """Per-residue 3Di logits predicted with ProstT5, kept in memory.

    Equivalent to loading the ``.npz`` that ``learnMSA-3di`` writes for the
    input sequences with its default settings, except that sequences share
    the logits of their representative in ``reuse``.
    """
    _require_torch_for_logits()
    from learnMSA.structure.io import tokens_as_logits
    from learnMSA.structure.prostt5 import (PROSTT5_CLASS_ORDER,
                                            predict_dataset)

    st = config.structure
    st.input_format = "logits"

    def predict(seqs: "SequenceDataset") -> "EmbeddingDataset":
        dataset = predict_dataset(seqs, verbose=config.input_output.verbose)
        _match_alphabet(dataset, PROSTT5_CLASS_ORDER,
                        st.structural_alphabet, "ProstT5")
        return dataset

    if reuse is None:
        return predict(data)
    # Residues without a counterpart in their representative carry no letter
    no_letter = tokens_as_logits(
        np.ones(len(st.structural_alphabet)),
        st.background_distribution,
        st.input_temperature,
    )
    return embed_representatives(
        data, reuse, lambda reps: predict(reps).embedding_cache, no_letter
    )


def find_plm_representatives(
    config: Configuration, data: "SequenceDataset"
) -> "RedundancyMap":
    """Groups sequences that share their language model embeddings, see
    ``language_model.plm_cluster_id``."""
    from learnMSA.util.redundancy import find_near_identical

    reuse = find_near_identical(
        data,
        config.language_model.plm_cluster_id,
        config.input_output.work_dir,
        verbose=config.input_output.verbose,
    )
    if config.input_output.verbose and not reuse.is_trivial:
        residues = data.seq_lens[reuse.rep_full_index].sum()
        print(
            f"Running language models on {reuse.num_rep} representatives of "
            f"{reuse.num_full} sequences "
            f"({residues / data.seq_lens.sum():.1%} of the residues)."
        )
    return reuse


def embed_representatives(
    data: "SequenceDataset",
    reuse: "RedundancyMap",
    embed: Callable[["SequenceDataset"], "EmbeddingCache"],
    placeholder: np.ndarray | None = None,
) -> "EmbeddingDataset":
    """Embeds only the representatives in ``reuse``; every other sequence
    shares the embeddings of its representative. Sequences in
    ``reuse.positions`` get a copy of the representative's embeddings at
    their aligned residues and ``placeholder`` at the others.

    Args:
        data: All sequences.
        reuse: Representatives of the sequences in data.
        embed: Computes the per-residue embeddings of a dataset.
        placeholder: Embedding of residues without a counterpart in their
            representative. Defaults to the mean embedding.

    Returns:
        Embeddings of all sequences in data, in the same order.
    """
    from learnMSA.util import EmbeddingCache, EmbeddingDataset

    reps = data if reuse.is_trivial else data.subset(reuse.rep_full_index)
    cache = embed(reps)
    cache_indices = reuse.rep_index
    if reuse.positions:
        members = np.array(sorted(reuse.positions))
        starts = cache.cum_lens[cache_indices[members]]
        rows = np.concatenate([
            np.where(pos >= 0, start + pos, -1)
            for start, pos in zip(starts, map(reuse.positions.get, members))
        ])
        extra = cache.cache[rows]
        if placeholder is None:
            placeholder = cache.cache.mean(axis=0, dtype=np.float32)
        extra[rows < 0] = placeholder
        cache_indices = cache_indices.copy()
        cache_indices[members] = cache.seq_lens.size + np.arange(members.size)
        cache = EmbeddingCache(
            np.concatenate([cache.seq_lens, data.seq_lens[members]]),
            cache.dim,
            cache=np.concatenate([cache.cache, extra]),
        )
    return EmbeddingDataset(
        embedding_cache=cache,
        seq_ids=list(data.seq_ids),
        cache_indices=cache_indices,
    )


def load_emb_data(
    config: Configuration,
    data: "SequenceDataset",
    stack: ExitStack | None = None,
) -> "EmbeddingDataset | None":
    if config.input_output.emb_file is not None:
        from learnMSA.util import EmbeddingDataset
        dataset = EmbeddingDataset(config.input_output.emb_file)
        if stack is None:
            emb_data = dataset
        else:
            emb_data = stack.enter_context(dataset)
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

