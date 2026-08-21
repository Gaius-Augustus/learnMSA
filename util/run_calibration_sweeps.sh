#!/usr/bin/env bash
# Calibration sweeps for util/calibrate_impl_factor.py.
#
#   util/run_calibration_sweeps.sh              # both, one after the other
#   util/run_calibration_sweeps.sh pytorch      # just one
#   util/run_calibration_sweeps.sh tensorflow
#
# Backends never run concurrently: they share the single GPU, and a second
# process on the card would corrupt every peak-memory measurement, which is the
# whole point of the exercise.
#
# The interpreter is given by absolute path on purpose. This shell's PATH leads
# with learnMSAdev2/bin, and `conda run -n learnMSAtorch python` does not
# override that -- it silently runs the TensorFlow env's python, and the probe
# children then all fail with "No GPU visible".
set -u
cd /home/felix/src/learnMSA
mkdir -p util/calib_logs

WORKLOADS=train,viterbi,posterior,loglik
FEATURES=aa,structure,language_model,both

run() {  # run <env> <backend> <tag>
    local env=$1 backend=$2 tag=$3
    {
        echo "=== $backend sweep starting $(date -Is) ==="
        conda run --live-stream -n "$env" \
            "/home/felix/miniforge3/envs/$env/bin/python" \
            util/calibrate_impl_factor.py \
                --backend "$backend" --compile off \
                --features "$FEATURES" --workloads "$WORKLOADS" \
                -o "util/impl_factor_calibration_rtx3090_$tag.json"
        echo "=== $backend sweep exited $? at $(date -Is) ==="
    } 2>&1 | tee "util/calib_logs/$tag.log"
}

case "${1:-all}" in
    pytorch)    run learnMSAtorch pytorch    torch ;;
    tensorflow) run learnMSAdev2  tensorflow tf    ;;
    all)        run learnMSAtorch pytorch    torch
                run learnMSAdev2  tensorflow tf    ;;
    *) echo "usage: $0 [pytorch|tensorflow|all]" >&2; exit 2 ;;
esac

echo "=== CALIBRATION DONE $(date -Is) ==="
