
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-optimized}"
BLOCK="${2:-16}"

echo "Generating figures for $MODE, block $BLOCK..."

python3 scripts/plot.py \
    --mode "$MODE" \
    --block "$BLOCK"

echo "Figures generated successfully."
