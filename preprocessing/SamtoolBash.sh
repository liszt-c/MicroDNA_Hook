#!/bin/bash
# check	samtools
which samtools &>/dev/null || { echo "samtools not found!"; exit 1; }

# end of checking

if [ $# -lt 6 ];then

echo "Need 6 parameters! <bedgraph> <chrom info>"

exit

fi

F=$1

G=$2

H=$3

Z=$4

N=$5

W=$6

samtools faidx ${F}.${W} ${G}:${H}-${Z} > chr${N}.fa