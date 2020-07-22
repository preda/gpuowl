#!/usr/bin/env bash

set -o xtrace

find $1 -name '*.proof' | xargs -I{} cp "{}" $2
