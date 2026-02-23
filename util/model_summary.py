#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf

from learnMSA.model.tf import LearnMSAModel as _LearnMSAModel
from learnMSA.model.tf.training import BatchGenerator
from learnMSA.util import SequenceDataset


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Load a learnMSA checkpoint, print transition/emission matrices, "
			"and run a one-sequence batch through the model."
		)
	)
	parser.add_argument(
		"checkpoint",
		type=Path,
		help="Path to a saved Keras checkpoint (.keras).",
	)
	parser.add_argument(
		"--fasta",
		type=Path,
		default=None,
		help=(
			"Optional FASTA file. If provided, the first sequence is used for "
			"batch generation."
		),
	)
	return parser.parse_args()


def load_first_sequence(fasta_path: Path) -> tuple[str, str]:
	with SequenceDataset(fasta_path, "fasta") as data:
		data.validate_dataset(single_seq_ok=True)
		seq_id = data.seq_ids[0]
		seq = data.get_standardized_seq(0)
	return seq_id, seq


def make_single_sequence_dataset(fasta_path: Path | None) -> SequenceDataset:
	if fasta_path is not None:
		seq_id, seq = load_first_sequence(fasta_path)
		return SequenceDataset(sequences=[(seq_id, seq)])
	return SequenceDataset(sequences=[("dummy_seq", "A")])


def print_model_matrices(model: tf.keras.Model) -> None:
	transitioner = model.phmm_layer.hmm.transitioner
	print("=== Transitioner Matrices ===")
	explicit_matrix = transitioner.explicit_transitioner.matrix().numpy()
	print(f"explicit_transitioner.matrix() shape: {explicit_matrix.shape}")
	print(explicit_matrix)

	folded_matrix = transitioner.matrix().numpy()
	print(f"transitioner.matrix() shape: {folded_matrix.shape}")
	print(folded_matrix)

	print("=== Emitter Matrices ===")
	for index, emitter in enumerate(model.phmm_layer.hmm.emitter):
		if not hasattr(emitter, "matrix"):
			print(f"emitter[{index}] ({type(emitter).__name__}): no matrix() method")
			continue
		try:
			matrix = emitter.matrix().numpy()
		except Exception as exc:
			print(
				f"emitter[{index}] ({type(emitter).__name__}): failed to compute "
				f"matrix() ({exc})"
			)
			continue
		print(f"emitter[{index}] ({type(emitter).__name__}) shape: {matrix.shape}")
		print(matrix)


def make_batch(model: tf.keras.Model, fasta_path: Path | None):
	single_data = make_single_sequence_dataset(fasta_path)
	batch_gen = BatchGenerator(static_shape_mode=True)
	batch_gen.configure(single_data, model.context)
	return batch_gen(np.arange(1))


def main() -> None:
	args = parse_args()

	print(f"Loading checkpoint: {args.checkpoint}")
	model = tf.keras.models.load_model(args.checkpoint)

	if args.fasta is not None:
		print(f"Using first sequence from FASTA: {args.fasta}")
	else:
		print("No FASTA provided. Using fallback one-residue sequence: 'A'.")

	print_model_matrices(model)

	print("=== Batch + Model Outputs ===")
	seq_batch = make_batch(model, args.fasta)
	batch_tensor, batch_indices = seq_batch
	print(f"batch tensor shape: {batch_tensor.shape}")
	print(batch_tensor)
	print(f"batch model-index mapping shape: {batch_indices.shape}")
	print(batch_indices)

	encoded = model.encode_batch(seq_batch)
	print(f"encoded batch shape: {encoded.shape}")
	print(encoded.numpy())

	model_output = model(seq_batch)
	print(f"model output shape: {model_output.shape}")
	print(model_output.numpy())


if __name__ == "__main__":
	main()
