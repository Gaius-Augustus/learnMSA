#!/usr/bin/env bash
# Sequential calibration sweeps, one backend after the other.
#
# They share the single GPU: running them concurrently would corrupt every
# peak-memory measurement, which is the whole point of the exercise.
#
# The interpreter is given by absolute path on purpose. This shell's PATH
# leads with learnMSAdev2/bin, and `conda run -n learnMSAtorch python` does
# not override that -- it silently runs the TensorFlow env's python.
set -u
cd /home/felix/src/learnMSA

WORKLOADS=train,viterbi,posterior,loglik
FEATURES=aa,structure,language_model,both

run() {  # run <env> <backend> <report>
    local env=$1 backend=$2 out=$3
    echo "=== $backend sweep starting $(date -Is) ==="
    conda run --live-stream -n "$env" \
        "/home/felix/miniforge3/envs/$env/bin/python" \
        util/calibrate_impl_factor.py \
            --backend "$backend" --compile off \
            --features "$FEATURES" --workloads "$WORKLOADS" \
            -o "$out"
    echo "=== $backend sweep exited $? at $(date -Is) ==="
}

run learnMSAtorch pytorch \
    util/impl_factor_calibration_rtx3090_torch.json \
    2>&1 | tee util/calib_logs/torch.log

run learnMSAdev2 tensorflow \
    util/impl_factor_calibration_rtx3090_tf.json \
    2>&1 | tee util/calib_logs/tf.log

echo "=== CALIBRATION DONE $(date -Is) ==="
