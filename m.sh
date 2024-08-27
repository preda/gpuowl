#!/usr/bin/bash

make DEBUG=1 -j4 && cp ./build-debug/prpll prpll-debug && make DEBUG=0 -j4 && mv ./build-release/prpll prpll-release
