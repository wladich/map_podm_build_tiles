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

if [ -e $BORDER_DIR ]; then
    pushd $BORDER_DIR
    git pull
    popd
else
    mkdir -p $BORDER_DIR
    git clone https://github.com/wladich/map_podm_border.git $BORDER_DIR
fi
