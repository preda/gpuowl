#!/usr/bin/env bash

ls -l $1 | grep .proof | wc -l
