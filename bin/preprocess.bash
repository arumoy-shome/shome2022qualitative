#!/usr/bin/env bash

# Lowercase the protected attribute field for the meps dataset.

[[ ! "$1" ]] && exit

sed -i '' 's/RACE/race/g' "$1"
