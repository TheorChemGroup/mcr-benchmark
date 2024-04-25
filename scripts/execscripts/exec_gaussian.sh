#!/bin/sh

cd "$1"

INPFILE=$2
GVER=g16

RUNF=`echo "$GVER" | cut -c 1-3`

#GVER:
# g09d - 64-bit Gaussian09 D-01
# g16sse - 64-bit Gaussian16 A-03 compiled with SSE4.2 instructions (older) and to use GPUs
# g16avx - 64-bit Gaussian16 A-03 compiled with AVX2 instructions (newer) and to use GPUs
# /lustre/opt/software/applied/

MSCR=/dev/shm/scratch_`whoami`_$3

rm -rf ${MSCR}

mkdir ${MSCR}
chmod 777 ${MSCR}

export GAUSS_SCRDIR=${MSCR}
export GAUSS_EXEDIR="/s/ls4/groups/g0130/bin/"$GVER""
export GAUSS_ARCHDIR="/"$GVER"/arch"
export GMAIN=$GAUSS_EXEDIR
export LD_LIBRARY_PATH=$GAUSS_EXEDIR
export G09BASIS="/s/ls4/groups/g0130/bin/"$GVER"/basis"
export F_ERROPT1="271,271,2,1,2,2,2,2"
export TRAP_FPE="OVERFL=ABORT;DIVZERO=ABORT;INT_OVERFL=ABORT"
export MP_STACK_OVERFLOW="OFF"
export KMP_DUPLICATE_LIB_OK="TRUE"
export PATH=${PATH}:${GAUSS_EXEXDIR}
/s/ls4/groups/g0130/bin/"$GVER"/"$RUNF" $INPFILE

rm -rf ${MSCR}
