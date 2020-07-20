import argparse
import os
import subprocess
import numpy as np
from shutil import copyfile
import open3d as o3
import time
from shutil import copyfile

# Adjust this to the current path
PROJECT_PATH = '/home/gustavo/fitler/'

# Path to MIT dataset with the .obj files [http://people.csail.mit.edu/drdaniel/mesh_animation/#data]
IC_DIR = PROJECT_PATH + 'segmented/mesh1/'

BINARY = PROJECT_PATH + 'build/apps/app/app'

# Set <CONFIG> parameters to use GT for metric
EVAL = PROJECT_PATH + 'build/apps/app/app -crispness"
CONFIG = PROJECT_PATH + "configs/config_mesh.json"
SEG_CONFIG = PROJECT_PATH + "configs/config_seg.json"
RESULT_FILE = PROJECT_PATH + 'dummy.pcd' 

# Name of the .csv generated file
METRIC_FILE = PROJECT_PATH + 'segmented/mesh_ours'

def evaluate_path(next_path):
    cmd = '%s -f1 %s -f2 %s -config %s -seg-config %s -result-file %s'%(EVAL, RESULT_FILE, next_path, CONFIG, SEG_CONFIG, METRIC_FILE)
    #subprocess.call(cmd, shell=True)
    print(cmd)
    p = subprocess.Popen(cmd, shell=True)
    p.wait()

def execute_reg(current_frame, next_frame):
    cmd = '%s -vec -f1 %s -f2 %s -config %s -seg-config %s -result-file %s'%(BINARY, current_frame, next_frame, CONFIG, SEG_CONFIG, RESULT_FILE)
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
    # copyfile(current_frame, RESULT_FILE)


def compare_frames(current_frame, next_frame):
    transf = execute_reg(current_frame, next_frame)
    evaluate_path(next_frame)

if __name__=="__main__":
    frames = []
    for subdir, dirs, files in os.walk(os.path.abspath(IC_DIR)):
        frames.extend(files)
    
    current_frame = frames[0]
    print(current_frame)
    current_frame_p = os.path.join(IC_DIR, current_frame)
    for frame in frames[1:]:
        print(frame)
        next_frame = frame
        next_frame_p = os.path.join(IC_DIR, next_frame)
        compare_frames(current_frame_p, next_frame_p)
        current_frame = next_frame
        current_frame_p = os.path.join(IC_DIR, current_frame)