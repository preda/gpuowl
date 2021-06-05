#!/usr/bin/bash

DIR=`dirname "$0"`
diff -y --suppress-common-lines <($DIR/counts.py < $1 | grep "Occupancy:") <($DIR/counts.py < $2 | grep "Occupancy:")
echo
echo ---------------------
echo

diff -y --suppress-common-lines <($DIR/counts.py < $1 | grep "NumVgprs:") <($DIR/counts.py < $2 | grep "NumVgprs:")
echo
echo ---------------------
echo

diff -y --suppress-common-lines <($DIR/counts.py < $1 | grep "ScratchSize:") <($DIR/counts.py < $2 | grep "ScratchSize:")
echo
echo ---------------------
echo

diff -y --suppress-common-lines <($DIR/counts.py < $1 | grep "codeLenInByte") <($DIR/counts.py < $2 | grep "codeLenInByte")
echo
echo ---------------------
echo
diff -y --suppress-common-lines <($DIR/counts.py < $1 | grep -v "codeLenInByte\|NumVgprs:\|Occupancy:\|ScratchSize:") <($DIR/counts.py < $2 | grep -v "codeLenInByte\|NumVgprs:\|Occupancy:\|ScratchSize:")
