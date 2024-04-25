#!/bin/sh

# Single-threaded execution of xtb
export OMP_NUM_THREADS=1

# Change to the specified directory
cd "$1"

# Run the xtb command with the appropriate input file and options, and redirect the output to a log file
/s/ls4/groups/g0130/knv_bin/xtb/build/xtb start.xyz --gff --opt tight --cycles 5000 -chrg $2 --alpb water > log
