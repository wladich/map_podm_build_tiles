#!/bin/bash
set -e
. config
if [ -e $VMAPS_DIR ]; then
    pushd $VMAPS_DIR;
    git pull
    popd
else
    mkdir -p $VMAPS_DIR
    git clone --depth=1 https://github.com/slazav/map_podm.git $VMAPS_DIR
fi


