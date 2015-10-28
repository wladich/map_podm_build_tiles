#!/bin/bash
set -e
. config

[ -e "$MBTILES_FILE" ] && rm "$MBTILES_FILE"
python plt2json.py $VMAPS_DIR/BRD/*.plt > $BORDER
nice python make_tiles.py --vmap $VMAPS_DIR/vmap --out $MBTILES_FILE --format mbtiles --rscale 50000 --max-level 14 --metatile-level 9 --border $BORDER --highlight-level=4
