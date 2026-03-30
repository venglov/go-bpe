#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/go/export"

OUT_DIR="../../python/go_bpe"

if [[ "$(uname)" == "Darwin" ]]; then
    EXT="dylib"
else
    EXT="so"
fi

CGO_ENABLED=1 go build -buildmode=c-shared \
    -o "${OUT_DIR}/libbpe.${EXT}" .

rm -f "${OUT_DIR}/libbpe.h"

echo "Built ${OUT_DIR}/libbpe.${EXT}"
