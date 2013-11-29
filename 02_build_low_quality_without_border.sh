#!/bin/bash
set -e
. config
mkdir -p $PREVIEW_TILES_DIR
rm -rf $PREVIEW_TILES_DIR/*

nice python make_tiles.py  --no-size-optimize --vmap $VMAPS_DIR/vmap --out $PREVIEW_TILES_DIR --low-quality --rscale 50000 --max-level 12 --meta-level 9