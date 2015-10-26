#!/bin/bash
set -e
. config

python plt2json.py $VMAPS_DIR/BRD/*.plt > $BORDER
firefox "editborder.html?tiles_dir=$PREVIEW_TILES_DIR&border=$BORDER"
