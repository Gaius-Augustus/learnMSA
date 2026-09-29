# Fit commands for the 3Di priors with a lower bound on the concentrations
# (every alpha > min-alpha > 1, so the MAP never pulls a letter to zero).
#
#   scop_3Di_a1p1_20_20_1   --min-alpha 1.1   --neff-prior-conc 20
#   scop_3Di_a1p01_20_20_1  --min-alpha 1.01  --neff-prior-conc 20
#   scop_3Di_a1p1_10_20_1   --min-alpha 1.1   --neff-prior-conc 10
#
# Data: SCOP2 superfamily 3Di alignments filtered by TM-score
# (~/data/SCOP/superfamily_3Di_filtered, see ~/data/SCOP/doc.md); same recipe
# as the scop_3Di_<conc>_20 priors. Run from the repository root in the
# TensorFlow environment (learnMSAdev2).
fit() {
    python -m learnMSA.hmm.tf.fit_dirichlet ~/data/SCOP/superfamily_3Di_filtered \
        --alphabet 3di --pattern '*.fasta' -c 1 --num-runs 5 --epochs 1000 \
        --min-count 3 --neff-prior-conc "$2" --min-alpha "$1" \
        -o "learnMSA/hmm/weights/$3_1.npz"
}
fit 1.1  20 scop_3Di_a1p1_20_20
fit 1.01 20 scop_3Di_a1p01_20_20
fit 1.1  10 scop_3Di_a1p1_10_20
