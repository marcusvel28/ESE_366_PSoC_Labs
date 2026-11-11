
#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "Generating FNN figures..."

python3 scripts/plot.py

echo "All figures generated successfully."
