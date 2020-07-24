#!/usr/bin/env bash

set -o xtrace

# Copy files to upload location.
find $1 -name '*.proof' -exec cp -v -n '{}' $2 \;

# Also copy to a backup location.
 [[ ! -z "$3" ]] && find $1 -name '*.proof' -exec cp -v -n '{}' $3 \; || echo "No backup location specified, proof files still copied to upload location."
 
