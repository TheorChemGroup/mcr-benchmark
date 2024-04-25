#!/bin/sh

module load cmake/3.19.3 gcc/10.2.0

# Single-threaded execution of xtb
export OMP_NUM_THREADS=1

# Check if the second argument exists
if [ -z "$3" ]; then
    # If the second argument is not provided, use "start.xyz"
    file="start.xyz"
else
    # If the second argument is provided, use it as the input file
    file="$3"
fi

# Change to the specified directory
cd "$1"

# Run the xtb command with the appropriate input file and options, and redirect the output to a log file
/s/ls4/groups/g0130/knv_bin/xtb/build/xtb "$file" --gff --md -chrg $2 --alpb water > log
