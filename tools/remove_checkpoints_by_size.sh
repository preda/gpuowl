#!/usr/bin/env bash

# This one-liner is still valid, however you have to pass the subdirectory with checkpoint files as first argument;
# In the subdirectory there are two checkpoint files: *.owl and *-old.owl
# 

set -o xtrace

find $1 -name '*.owl' -a -size 0 -exec rm -v {} + | tee -a ./removed_checkpoints.log
