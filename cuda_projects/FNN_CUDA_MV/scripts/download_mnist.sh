#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/mnist
cd data/mnist
BASE="https://storage.googleapis.com/cvdf-datasets/mnist"
for file in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
    curl -fL --retry 3 -o "$file" "$BASE/$file"
    gzip -dkf "$file"
done
ls -lh *ubyte
