#!/usr/bin/env bash
# Downloads and extracts the RealWaste dataset into data/RealWaste/.
# Idempotent: safe to re-run, skips work if the data is already present.
#
# Source: UCI Machine Learning Repository (CC BY 4.0).
# Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste [Dataset].
# https://doi.org/10.24432/C5SS4G
#
# NOTE: this is a ~657MB download (vs. TrashNet's 43MB) — real photos
# at 524x524, not the tiny curated set we started with. Give it a minute.
set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
ZIP_PATH="$DATA_DIR/realwaste.zip"
URL="https://archive.ics.uci.edu/static/public/908/realwaste.zip"

if [ -d "$DATA_DIR/RealWaste" ]; then
  echo "RealWaste already present at $DATA_DIR/RealWaste, skipping download."
  exit 0
fi

mkdir -p "$DATA_DIR"
echo "Downloading RealWaste dataset (657MB, may take a minute)..."
curl -sL -o "$ZIP_PATH" "$URL"

echo "Extracting..."
unzip -q "$ZIP_PATH" -d "$DATA_DIR"
rm -f "$ZIP_PATH"

# The zip extracts to a nested "realwaste-main/RealWaste/" structure;
# flatten it so data/RealWaste/<class>/ is directly usable.
if [ -d "$DATA_DIR/realwaste-main/RealWaste" ]; then
  mv "$DATA_DIR/realwaste-main/RealWaste" "$DATA_DIR/RealWaste"
  rm -rf "$DATA_DIR/realwaste-main"
fi

echo "Done. $(find "$DATA_DIR/RealWaste" -type f | wc -l) images in $DATA_DIR/RealWaste"
