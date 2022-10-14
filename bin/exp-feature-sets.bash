#!/usr/bin/env bash

# Parallel execution of bin/exp-feature-sets.bash
#
# This script executes the bin/exp-feature-sets.py script in parallel
# for all dataset & protected attribute pairs. The script accepts the
# number of iterations as a positional argument. The default is set to
# 1 iteration.

# Usage:
# To run exp-feature-sets.py for all datasets & 5 iterations, run the
# following command:
#
#    ./bin/exp-feature-sets.bash 5
#

# This script starts as many processes as possible by default. You can
# adjust this number by changing the value passed to the -P flag
# below. Consult the man pages for xargs for more information.

ITERATIONS=1
[[ "$1" ]] && ITERATIONS="$1"

DATASETS=(
    "adult-race $ITERATIONS"
    "adult-sex $ITERATIONS"
    "compas-race $ITERATIONS"
    "compas-sex $ITERATIONS"
    "bank-age $ITERATIONS"
    "german-sex $ITERATIONS"
    "german-age $ITERATIONS"
    "meps-RACE $ITERATIONS"
)

echo "${DATASETS[@]}" |
    xargs -n 2 -P 0 .venv/bin/python3 bin/exp-feature-sets.py

