#!/usr/bin/env bash

DATASETS=(
    "adult-race"
    "adult-sex"
    "compas-race"
    "compas-sex"
    "bank-age"
    "german-sex"
    "german-age"
    "meps-RACE"
)

echo "${DATASETS[@]}" |
    xargs -n 1 -P 0 python3 bin/visualise-exp-feature-sets.py

