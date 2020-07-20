#!/bin/bash

X="/mnt/c/Users/gusta/Documents/Research/MR-RPM/X.txt"
Y="/mnt/c/Users/gusta/Documents/Research/MR-RPM/Y.txt"
Z="/mnt/c/Users/gusta/Documents/Research/MR-RPM/Z.txt"

#CONFIG="/home/gustavo/fitler/mr_rpm/config_mrrpm.json"
CONFIG="/home/gustavo/fitler/config_horse.json"
SEG_CONFIG="/home/gustavo/fitler/config_seg.json"

RESULT_DIR="/mnt/c/Users/gusta/Documents/Research/MR-RPM/RESULT"

# Custom code to account for 3D feature alignment
/home/gustavo/fitler/build/apps/mr_rpm/mr_rpm -mrrpm -X $X -Y $Y -Z $Z -config $CONFIG -seg-config $SEG_CONFIG -result-file $RESULT_DIR