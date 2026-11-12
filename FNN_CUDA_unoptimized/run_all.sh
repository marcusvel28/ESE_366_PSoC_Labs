#!/bin/bash
# Builds and runs 3 variants x 3 block sizes and saves each run's output to results/.
#
#   optimized      tiled shared-memory kernels + fused operations      (nn_project/)
#   naive_L1       global-memory-only kernels, L1 caches global loads  (nn_project_naive/)
#   naive_noL1     global-memory-only kernels, global loads skip L1    (nn_project_naive/ + -dlcm=cg)
#
# Run it from the folder that contains nn_project/, nn_project_naive/ and data/MNIST/raw/.
# Before timing on the Jetson:  sudo nvpmodel -m 0 && sudo jetson_clocks
set -e
ARCH=sm_87
UNROLL=4
mkdir -p results

build_and_run() {   # $1 = label, $2 = source folder, $3 = block size, $4... = extra nvcc flags
    label=$1; dir=$2; bs=$3; shift 3
    exe=./nn_${label}_${bs}
    echo "=== $label, BLOCK_SIZE=$bs ==="
    nvcc -O3 -arch=$ARCH -DBLOCK_SIZE=$bs -DUNROLL=$UNROLL "$@" -o $exe \
        $dir/main.cu $dir/neural_network.cu $dir/matrix_operations.cu
    $exe | tee results/${label}_${bs}.txt
}

for bs in 8 16 32; do
    build_and_run optimized  nn_project       $bs
    build_and_run naive_L1   nn_project_naive $bs
    build_and_run naive_noL1 nn_project_naive $bs -Xptxas -dlcm=cg
done

echo
echo "Done. Per-run output is in results/. Compare the 'train time' column across files."
