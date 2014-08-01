#!/bin/bash
set -e
. config

nice python make_tiles.py --vmap $VMAPS_DIR/vmap --out $MBTILES_FILE --format mbtiles --rscale 50000 --max-level 14 --metatile-level 9 --border $BORDER_DIR/border.txt