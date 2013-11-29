#!/bin/bash
set -e
. config
mkdir -p $TILES_DIR
rm -rf $TILES_DIR/*

nice python make_tiles.py  --vmap $VMAPS_DIR/vmap --out $TILES_DIR --rscale 50000 --max-level 14 --meta-level 9 --border $BORDER_DIR/border.txt