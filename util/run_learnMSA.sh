#!/bin/bash
#SBATCH --job-name=learnMSA
#SBATCH --output=slurm/learnMSA%j.out
#SBATCH --error=slurm/learnMSA%j.err
#SBATCH --nodes=1
#SBATCH --partition=vision
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=100gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8


learnMSA \
-i ../test/data/egf.fasta \
-o ../test/egf.out \
--use_language_model \
--sequence_weights
