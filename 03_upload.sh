#!/bin/bash

set -e
set -x
DIR=`dirname $0`
DIR=`readlink -f $DIR`

. $DIR/config

rsync -P "$WORKDIR/$MBTILES_FILE" root@${server}:${upload_path}

