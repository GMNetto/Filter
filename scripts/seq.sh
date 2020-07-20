#!/bin/bash

#TRACKLET="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0091_tracklets/2011_09_26/2011_09_26_drive_0091_sync/tracklet_labels.xml"
#DIR="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0091_sync/2011_09_26/2011_09_26_drive_0091_sync/velodyne_points/data/"
TRACKLET="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0005_tracklets/2011_09_26/2011_09_26_drive_0005_sync/tracklet_labels.xml"
DIR="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/"
#TRACKLET="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0060_tracklets/2011_09_26/2011_09_26_drive_0060_sync/tracklet_labels.xml"
#DIR="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0060_sync/2011_09_26/2011_09_26_drive_0060_sync/velodyne_points/data/"
#TRACKLET="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0039_tracklets/2011_09_26/2011_09_26_drive_0039_sync/tracklet_labels.xml"
#DIR="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0039_sync/2011_09_26/2011_09_26_drive_0039_sync/velodyne_points/data/"
#TRACKLET="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0020_tracklets/2011_09_26/2011_09_26_drive_0020_sync/tracklet_labels.xml"
#DIR="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0020_sync/2011_09_26/2011_09_26_drive_0020_sync/velodyne_points/data/"

CLOUD1="/home/gustavo/fitler/segmented/05/4/2.txt"
CLOUD2="/home/gustavo/fitler/segmented/05/3/groundless.txt"
#CLOUD1="/home/gustavo/fitler/segmented/91/88/25.txt"
#CLOUD2="/home/gustavo/fitler/segmented/91/89/25.txt"

#CLOUD2="/home/gustavo/fitler/build/frame2/object1.pcd"

#CLOUD1="/mnt/c/Users/gusta/Desktop/testKitti/horse1.obj"
#CLOUD2="/mnt/c/Users/gusta/Desktop/testKitti/horse3.obj"

#CLOUD1="/mnt/c/Users/gusta/Downloads/meshes/meshes/meshes/mesh_0016.obj"
#CLOUD2="/mnt/c/Users/gusta/Downloads/meshes/meshes/meshes/mesh_0018.obj"

START=1
END=2
CONFIG="../configs/config_seq05.json"
SEG_CONFIG="../configs/config_seg.json"
TEMP_CONFIG="../temp_config.json"
#RESULT_FILE="/mnt/c/Users/gusta/Documents/Research/Results/"
#RESULT_FILE="/mnt/c/Users/gusta/Pictures/Results/Article3/scene/pm_icp.pcd"
RESULT_FILE="/mnt/c/Users/gusta/Pictures/Results/Article2/Intro2/icp_objects.pcd"
#RESULT_FILE="/home/gustavo/fitler/dummy.pcd"

#TRACKLET="/mnt/c/Users/gusta/Desktop/testKitti/2011_09_26_drive_0005_tracklets/2011_09_26/2011_09_26_drive_0005_sync/tracklet_labels.xml"
#DIR="/home/gustavo/fitler/segmented/91/"
#RESULT_FILE="/home/gustavo/fitler/segmented/91/"

#./apps/app/app -cubes -t $TRACKLET -dir $DIR -start $START -end $END -config $CONFIG -seg-config $SEG_CONFIG -result-file $RESULT_FILE -temp_config $TEMP_CONFIG
#./trans1 -crispness -f1 $CLOUD1 -f2 $CLOUD2 -config $CONFIG -seg-config $SEG_CONFIG -result-file $RESULT_FILE

./apps/app/app -vec -f1 $CLOUD1 -f2 $CLOUD2 -config $CONFIG -seg-config $SEG_CONFIG -result-file $RESULT_FILE