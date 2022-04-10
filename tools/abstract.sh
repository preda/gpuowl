#!/usr/bin/bash
DIR=`dirname "$0"`

diff <($DIR/abstract.py < $1) <($DIR/abstract.py < $2)
