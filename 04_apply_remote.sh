#!/bin/bash

set -e
set -x
DIR=`dirname $0`
DIR=`readlink -f $DIR`

. $DIR/config

ssh root@${server} "mv ${upload_path} ${target_path}"

