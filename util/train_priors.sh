#!/bin/bash

NEFF_CONC_VALUES=(2 2.25 2.5 2.75 3 3.25 3.5 3.75 4)

for conc in "${NEFF_CONC_VALUES[@]}"; do
    python learnMSA/hmm/tf/fit_dirichlet.py ~/data/PFAM/seed_fasta/ \
    --alphabet aa --name pfam_aa_neff_conc_${conc} --pattern *.fasta \
    --num-runs 10 --clans ~/data/PFAM/Pfam-A.clans.tsv -c 1 \
    --neff-prior-conc ${conc} --extended-alphabet --epochs 200
done
