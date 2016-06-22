#!/bin/bash

set -e
set -x
DIR=`dirname $0`
DIR=`readlink -f $DIR`

. $DIR/config

server=vultr.wladich.tk
target_path=/var/www/tiles/layers/map_podm_new.mb

scp "$WORKDIR/$MBTILES_FILE" root@${server}:${target_path}

