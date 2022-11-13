#!/bin/bash

#TRACKLET="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0091_tracklets/2011_09_26/2011_09_26_drive_0091_sync/tracklet_labels.xml"
#DIR="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0091_sync/2011_09_26/2011_09_26_drive_0091_sync/velodyne_points/data/"
TRACKLET="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0005_tracklets/2011_09_26/2011_09_26_drive_0005_sync/tracklet_labels.xml"
DIR="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/"
#TRACKLET="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0060_tracklets/2011_09_26/2011_09_26_drive_0060_sync/tracklet_labels.xml"
#DIR="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0060_sync/2011_09_26/2011_09_26_drive_0060_sync/velodyne_points/data/"
#DIR="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0060_sync/2011_09_26/2011_09_26_drive_0060_sync/velodyne_points/data/"
#TRACKLET="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0039_tracklets/2011_09_26/2011_09_26_drive_0039_sync/tracklet_labels.xml"
#DIR="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0039_sync/2011_09_26/2011_09_26_drive_0039_sync/velodyne_points/data/"
#TRACKLET="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0020_tracklets/2011_09_26/2011_09_26_drive_0020_sync/tracklet_labels.xml"
#DIR="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0020_sync/2011_09_26/2011_09_26_drive_0020_sync/velodyne_points/data/"

CLOUD1="/mnt/c/Users/gusta/Pictures/Results/Article2/cyclist/1.txt"
#CLOUD2="/home/gustavo/filter/segmented/pedestrian_91/263/groundless.txt"
CLOUD2="/mnt/c/Users/gusta/Pictures/Results/Article2/cyclist/next_1.txt"
CLOUD1="/home/gustavo/filter/bcpd/ex_cloud1.txt"
CLOUD2="/home/gustavo/filter/segmented/91/28/groundless.txt"

#CLOUD2="/home/gustavo/fitler/build/frame2/object1.pcd"

#CLOUD1="/mnt/c/Users/gusta/Desktop/testKitti/horse1.obj"
#CLOUD2="/mnt/c/Users/gusta/Desktop/testKitti/horse3.obj"

CLOUD1="/home/gustavo/filter/segmented/pedestrian_05/4/groundless.txt"
CLOUD2="/home/gustavo/filter/segmented/pedestrian_05/5/groundless.txt"
# CLOUD1="/home/gustavo/filter/build/src.txt"
# CLOUD2="/home/gustavo/filter/build/tgt.txt"

#CLOUD1="/home/gustavo/filter/segmented/pedestrian_91/262/46.txt"
#CLOUD2="/mnt/c/Users/gusta/Documents/bcpd/clouds/test_combined.txt"

START=0
END=150
CONFIG="../configs/config_perm.json"
SEG_CONFIG="../configs/config_seg.json"
TEMP_CONFIG="../temp_config.json"
#RESULT_FILE="/mnt/c/Users/gusta/Documents/Research/Results/"
#RESULT_FILE="/mnt/c/Users/gusta/Pictures/Results/Article3/scene/pm_icp.pcd"
#RESULT_FILE="/mnt/c/Users/gusta/Pictures/Results/Article2/Intro2/icp_objects.pcd"
#RESULT_FILE="/mnt/c/Users/gusta/Pictures/Results/Article2/cyclist/cyclist_shot.txt"
RESULT_FILE="/home/gustavo/filter/dummy.txt"
#"/home/gustavo/filter/cpd/sampled.txt"

#TRACKLET="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0005_tracklets/2011_09_26/2011_09_26_drive_0005_sync/tracklet_labels.xml"
#DIR="/home/gustavo/filter/segmented/pedestrian_91"
#RESULT_FILE="/mnt/c/Users/gusta/Pictures/Results/Article2/cyclist/new_ours.txt"

#./apps/app/app -cubes -t $TRACKLET -dir $DIR -start $START -end $END -config $CONFIG -seg-config $SEG_CONFIG -result-file $RESULT_FILE -temp_config $TEMP_CONFIG
#./trans1 -crispness -f1 $CLOUD1 -f2 $CLOUD2 -config $CONFIG -seg-config $SEG_CONFIG -result-file $RESULT_FILE

#./apps/app/app -sample -f1 $CLOUD1 -config $CONFIG -seg-config $SEG_CONFIG -result-file $RESULT_FILE
./apps/app/app -color -f1 $CLOUD1 -f2 $CLOUD2 -config $CONFIG -seg-config $SEG_CONFIG -result-file $RESULT_FILE