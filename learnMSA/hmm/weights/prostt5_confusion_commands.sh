# ProstT5 3Di observation matrix P(predicted | true) from Homstrad pairs:
# true 3Di = Foldseek on the Homstrad PDB files, predicted = foldseek
# createdb --prostt5-model (prostt5-f16.gguf) on the Homfam sequences.
python util/fit_3di_confusion.py \
    --true-dir ~/src/snakeMSA/data/homstrad/3Di \
    --pred-dir ~/src/snakeMSA/data/homfam/predicted_3di \
    --pseudocount 1 --bootstrap 200 \
    --out learnMSA/hmm/weights/prostt5_3Di_confusion_homstrad.npz
