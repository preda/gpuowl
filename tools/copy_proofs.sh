#!/usr/bin/env bash

if [[ $# -eq 2 ]]; then
  if [[ "$1" == "$2" ]]; then
    echo "Cannot copy to the same directory.";
    exit;
  fi
  # Copy files to backup location.
  find $1 -name '*.proof' -exec cp -v -n '{}' $2 \;
  exit;
elif [[ $# -lt 2 ]]; then
  echo "Too few arguments. Must specify original and destination directories.";
elif [[ $# -gt 2 ]]; then
  echo "Too much arguments. Must specify original and destination directories only.";
fi
