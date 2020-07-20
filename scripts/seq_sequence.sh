#!/bin/bash


START=1
END=2
CONFIG="../configs/config_seq91.json"
SEG_CONFIG="../configs/config_seg.json"
TEMP_CONFIG="../temp_config.json"

TRACKLET="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0005_tracklets/2011_09_26/2011_09_26_drive_0005_sync/tracklet_labels.xml"
# Assume files already segmented
DIR="/home/gustavo/fitler/segmented/91/"
RESULT_FILE="/home/gustavo/fitler/segmented/91/"

./apps/app/app -cubes -t $TRACKLET -dir $DIR -start $START -end $END -config $CONFIG -seg-config $SEG_CONFIG -result-file $RESULT_FILE -temp_config $TEMP_CONFIG
