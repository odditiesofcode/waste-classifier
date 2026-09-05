#!/usr/bin/env bash
# Downloads and extracts the TrashNet dataset into data/dataset-resized/.
# Idempotent: safe to re-run, skips work if the data is already present.
set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
ZIP_PATH="$DATA_DIR/dataset-resized.zip"
URL="https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"

if [ -d "$DATA_DIR/dataset-resized" ]; then
  echo "Dataset already present at $DATA_DIR/dataset-resized, skipping download."
  exit 0
fi

mkdir -p "$DATA_DIR"
echo "Downloading TrashNet dataset..."
curl -sL -o "$ZIP_PATH" "$URL"

echo "Extracting..."
unzip -q "$ZIP_PATH" -d "$DATA_DIR"
rm -rf "$DATA_DIR/__MACOSX" "$ZIP_PATH"

echo "Done. $(find "$DATA_DIR/dataset-resized" -type f | wc -l) images in $DATA_DIR/dataset-resized"
