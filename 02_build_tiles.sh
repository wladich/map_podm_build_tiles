#!/bin/bash
set -e
set -x
DIR=`dirname $0`
DIR=`readlink -f $DIR`

. $DIR/config

[ -e "$WORKDIR/$MBTILES_FILE" ] && rm "$WORKDIR/$MBTILES_FILE"
python $DIR/plt2json.py $WORKDIR/vmaps/BRD/*.plt > $WORKDIR/border.txt
docker run --rm -v $WORKDIR:/maps -v $DIR/make_tiles.py:/make_tiles.py:ro slazav_tiles bash -c "
nice python /make_tiles.py --vmap /maps/vmaps/vmap --out /maps/$MBTILES_FILE --format mbtiles --border /maps/border.txt $RENDER_ARGS
"
