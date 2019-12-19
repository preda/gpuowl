#!/usr/bin/bash

DIR=`dirname "$0"`
diff -y --suppress-common-lines <($DIR/counts.py < $1/*.s) <($DIR/counts.py < $2/*.s)
