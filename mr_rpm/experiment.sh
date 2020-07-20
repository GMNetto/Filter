#!/bin/bash

C1="/mnt/c/Users/gusta/Documents/Research/MR-RPM/X.txt"
#C2="/mnt/c/Users/gusta/Documents/Research/MR-RPM/experiments/density/person2/next/43.txt"
C2="/mnt/c/Users/gusta/Desktop/testKitti/horse2.txt"

#CONFIG="/home/gustavo/fitler/mr_rpm/config_mrrpm.json"
CONFIG="/home/gustavo/fitler/config_horse.json"
SEG_CONFIG="/home/gustavo/fitler/config_seg.json"

RESULT_FILE="/mnt/c/Users/gusta/Documents/Research/MR-RPM/RESULT"

/home/gustavo/fitler/build/apps/app/app -crispness -f1 $C1 -f2 $C2 -config $CONFIG -seg-config $SEG_CONFIG -result-file $RESULT_FILE
