#!/bin/sh

cd $1
export OMP_NUM_THREADS=1
/s/ls4/groups/g0130/knv_bin/crest/build/crest start.xyz --gfnff -xnam /s/ls4/groups/g0130/knv_bin/xtb/build/xtb > log
