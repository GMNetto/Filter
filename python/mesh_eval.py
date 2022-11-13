import argparse
import os
import subprocess
import numpy as np
from shutil import copyfile
import open3d as o3
import time
from shutil import copyfile

# Adjust this to the current path
PROJECT_PATH = '/home/gustavo/filter/'

CLOUD1 = PROJECT_PATH + "segmented/mesh1/mesh_0115.obj"
CLOUD2 = PROJECT_PATH + "segmented/mesh1/mesh_0116.obj"

# Path to MIT dataset with the .obj files [http://people.csail.mit.edu/drdaniel/mesh_animation/#data]
IC_DIR = PROJECT_PATH + 'segmented/mesh1/'

BINARY = PROJECT_PATH + 'build/apps/app/app'

# Set <CONFIG> parameters to use GT for metric
EVAL = PROJECT_PATH + 'build/apps/app/app -crispness'
CONFIG = PROJECT_PATH + "configs/mesh/config_mesh_perm.json"
SEG_CONFIG = PROJECT_PATH + "configs/config_seg.json"
RESULT_FILE = PROJECT_PATH + 'dummy.pcd' 

RESULT_SAMPLE = PROJECT_PATH + 'keypoints.txt' 

# Name of the .csv generated file
METRIC_FILE = PROJECT_PATH + 'python/mesh_'

def r_diff(cloud1, cloud2):
    diff = cloud2 - cloud1
    return np.sqrt(np.trace(np.dot(diff.T, diff))/cloud1.shape[0])
    
def error_function(trans, cloud1, cloud2):
    trans_r = r_diff(trans, cloud2)
    trans_o = r_diff(cloud1, cloud2)
    return 1 - trans_r/trans_o
    

def evaluate_path(current_path, next_path, mode):
    current_cloud = np.loadtxt(current_path)
    next_cloud = np.loadtxt(next_path)
    trans_cloud = np.loadtxt(RESULT_FILE)
    print(current_path, next_path)
    error = error_function(trans_cloud, current_cloud, next_cloud)
    print("ERROR", error)
    with open(METRIC_FILE + mode + '.txt', 'a+') as f:
        f.write("%.4f\n"%(error))
    

def execute_reg(current_frame, next_frame):
    cmd = '%s -vec -f1 %s -f2 %s -config %s -seg-config %s -result-file %s'%(BINARY, current_frame, next_frame, CONFIG, SEG_CONFIG, RESULT_FILE)
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
    # copyfile(current_frame, RESULT_FILE)


def compare_frames(current_frame, next_frame, mode):
    transf = execute_reg(current_frame, next_frame)
    evaluate_path(current_frame, next_frame, mode)

def get_keypoints(current_frame, next_frame):
    cmd = '%s -sample -f1 %s -config %s -seg-config %s -result-file %s'%(BINARY, current_frame, CONFIG, SEG_CONFIG, RESULT_SAMPLE)
    p = subprocess.Popen(cmd, shell=True)
    p.wait()

TWO_DIR_BINARY = "/usr/bin/python3 /home/gustavo/filter/python/two_dir_filter_upsample.py"
def reg_two_dir(current_frame, next_frame_p):
    flow = current_frame
    cmd = f'{TWO_DIR_BINARY} --cloud1 {current_frame} --cloud2 {next_frame_p} --flow {flow} --sigma2=0.1 --sigma2_f=0.1 --iter=10 --result {RESULT_FILE}'
    print(cmd)
    subprocess.run(cmd, shell=True)

def compare_two_dir(current_frame, next_frame):
    transf = reg_two_dir(current_frame, next_frame)
    evaluate_path(current_frame, next_frame, "two_dir")

transformed = 'output_y.txt'
BCPD_BINARY = '/mnt/c/Users/gusta/Documents/bcpd/win/bcpd.exe'
def reg_bcpd(current_frame, next_frame):
    cmd = '%s -x%s -y%s -w0.2 -b2.0 -l50 -g1.0 -J300 -K70 -c1e-6 -n150 -p -L100 '%(BCPD_BINARY, next_frame, current_frame)
    subprocess.call(cmd, shell=True)
    a = np.loadtxt(transformed)
    return a

def filter_bcpd(current_frame, next_frame):
    cmd = '%s -bcpd -f1 %s -f2 %s -f3 %s -f4 %s -config %s -seg-config %s -result-file %s'%(BINARY, RESULT_SAMPLE, 'result.txt', current_frame, next_frame, CONFIG, SEG_CONFIG, RESULT_FILE)
    p = subprocess.Popen(cmd, shell=True)
    p.wait()

def compare_bcpd(current_frame, next_frame):
    #get_keypoints(current_frame, next_frame)

    a = np.loadtxt(current_frame)
    comma_path = current_frame[:-4] + "_comma.txt"
    np.savetxt(comma_path, a, fmt='%.4f,%.4f,%.4f')

    comma_path2 = next_frame[:-4] + "_comma.txt"
    
    transf = reg_bcpd(comma_path, comma_path2)
    np.savetxt(RESULT_FILE, transf, fmt='%.4f %.4f %.4f')

    #filter_bcpd(current_frame, next_frame)
    evaluate_path(current_frame, next_frame, "bcpd")

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', type=str, default='ours',
     help='mode')
    return parser

if __name__=="__main__":
    args = init_parser().parse_args()
    frames = []
    for subdir, dirs, files in os.walk(os.path.abspath(IC_DIR)):
        frames.extend(files)
    
    current_frame = frames[0]
    print(current_frame)
    current_frame_p = os.path.join(IC_DIR, current_frame)
    skip = True
    if (args.mode == "single"):
        #compare_bcpd(CLOUD1, CLOUD2)
        compare_two_dir(CLOUD1, CLOUD2)
        #compare_frames(CLOUD1, CLOUD2)
        exit()
    for frame in frames[1:]:
        if frame[-10:] == '_comma.txt':
            continue
        if frame[-4:] != '.obj':
            continue
        # if skip:
        #     skip = False
        #     continue
        # skip = True
        print(frame)
        next_frame = frame
        next_frame_p = os.path.join(IC_DIR, next_frame)
        if (args.mode == 'bcpd'):
            compare_bcpd(current_frame_p, next_frame_p)
        elif args.mode == 'two_dir':
            compare_two_dir(current_frame_p, next_frame_p)
        else:
            compare_frames(current_frame_p, next_frame_p, args.mode)
        current_frame = next_frame
        current_frame_p = os.path.join(IC_DIR, current_frame)

