#!/usr/bin/env bash

# Parallel execution of bin/exp-training-sets.bash
#
# This script executes the bin/exp-feature-sets.py script in parallel
# for all dataset & protected attribute pairs. The script accepts the
# number of iterations as a positional argument. The default is set to
# 1 iteration.

# Usage:
# To run exp-training-sets.py for all datasets & 5 iterations, run the
# following command:
#
#    ./bin/exp-training-sets.bash 5
#

# This script is intended to be run within a docker container (see the
# readme for instructions on how to setup docker). It assumes that the
# number of cpus available to the container is restricted by the user
# thus it starts as many processes as possible by default. You can
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
    xargs -n 2 -P 0 python3 bin/exp-training-sets.py

