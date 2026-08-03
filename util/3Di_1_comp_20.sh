python learnMSA/hmm/tf/fit_dirichlet.py ~/src/snakeMSA/data/homstrad/aligned_3Di --alphabet 3di \
 --name homstrad_3Di_1_20 --pattern *.fasta --num-runs 5 \
 -c 1 --neff-prior-conc 1 --epochs 1000 --verbose --min-count 3

 python learnMSA/hmm/tf/fit_dirichlet.py ~/src/snakeMSA/data/homstrad/aligned_3Di --alphabet 3di \
 --name homstrad_3Di_3_20 --pattern *.fasta --num-runs 5 \
 -c 1 --neff-prior-conc 3 --epochs 1000 --verbose --min-count 3

 python learnMSA/hmm/tf/fit_dirichlet.py ~/src/snakeMSA/data/homstrad/aligned_3Di --alphabet 3di \
 --name homstrad_3Di_5_20 --pattern *.fasta --num-runs 5 \
 -c 1 --neff-prior-conc 5 --epochs 1000 --verbose --min-count 3

 python learnMSA/hmm/tf/fit_dirichlet.py ~/src/snakeMSA/data/homstrad/aligned_3Di --alphabet 3di \
 --name homstrad_3Di_10_20 --pattern *.fasta --num-runs 5 \
 -c 1 --neff-prior-conc 10 --epochs 1000 --verbose --min-count 3

 python learnMSA/hmm/tf/fit_dirichlet.py ~/src/snakeMSA/data/homstrad/aligned_3Di --alphabet 3di \
 --name homstrad_3Di_20_20 --pattern *.fasta --num-runs 5 \
 -c 1 --neff-prior-conc 20 --epochs 1000 --verbose --min-count 3