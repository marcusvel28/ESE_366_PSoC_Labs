#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

make

for mode in baseline optimized; do
    for block in 8 16 32; do
        echo "================================"
        echo "Testing $mode with block $block"
        echo "================================"

        ./fnn "$mode" "$block"

        bash scripts/plot.sh "$mode" "$block"
    done
done

echo "All experiments completed!"
