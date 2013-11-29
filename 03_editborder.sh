#!/bin/bash
set -e
. config

firefox "editborder.html?tiles_dir=$PREVIEW_TILES_DIR&border=$BORDER_DIR/border.txt"
