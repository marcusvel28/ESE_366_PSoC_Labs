
#!/usr/bin/env bash
set -euo pipefail

# Navigate to project root
cd "$(dirname "$0")/.."

# Generate plots for all CUDA configurations
for mode in baseline optimized; do
    for block in 8 16 32; do

        echo "================================"
        echo "Generating plots: $mode, block $block"
        echo "================================"

        # Skip configurations without training results
        if [ ! -f "outputs/$mode/block_$block/metrics.csv" ]; then
            echo "No results found. Skipping..."
            continue
        fi

        bash scripts/plot.sh "$mode" "$block"

    done
done

echo "All available plots generated successfully!"
