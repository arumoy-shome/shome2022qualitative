#!/usr/bin/env bash

if [[ "$1" ]]; then
    EXP="$1"
else
    echo "Error: positional argument not specified."
    exit 1
fi
DATASETS=(
    "adult-race $EXP"
    "adult-sex $EXP"
    "compas-race $EXP"
    "compas-sex $EXP"
    "bank-age $EXP"
    "german-sex $EXP"
    "german-age $EXP"
    "meps-RACE $EXP"
)

echo "${DATASETS[@]}" |
    xargs -n 2 -P 0 python3 bin/visualise-di-spd.py

