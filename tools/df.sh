#!/usr/bin/bash

DIR=`dirname "$0"`
# diff -y --suppress-common-lines <($DIR/counts.py < $1 | grep "Occupancy:") <($DIR/counts.py < $2 | grep "Occupancy:")
diff <($DIR/abstract.py < $1) <($DIR/abstract.py < $2)
