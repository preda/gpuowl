#!/usr/bin/env bash

set -o xtrace

find $1 -name '*.proof' -exec cp -v -n '{}' $2 \;
