#!/bin/bash
set -e
set -x
DIR=`dirname $0`
DIR=`readlink -f $DIR`

. $DIR/config

if [ -e $WORKDIR/vmaps ]; then
    pushd $WORKDIR/vmaps
    git pull
    popd
else
    mkdir -p $WORKDIR/vmaps
    git clone --depth=1 "$REPO_URL" $WORKDIR/vmaps
fi


