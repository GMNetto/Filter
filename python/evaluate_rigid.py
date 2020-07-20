import os
import argparse
import subprocess
import re
import math
import numpy as np

PROJECT_PATH = '/home/gustavo/fitler/'

CONFIG = PROJECT_PATH + "/configs/config_seq91.json"
SEG_CONFIG = PROJECT_PATH + '/configs/config_seg.json"
TRACKLET = PROJECT_PATH + '/segmented/91/tracklet_labels.xml'
DIR = PROJECT_PATH + "/segmented/91/"
RESULT_FILE = PROJECT_PATH + "/segmented/91/"
EVAL = PROJECT_PATH + "/build/apps/match/match -match"
METRIC_DIR = './'


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', type=str, default='',
     help='mode')
    parser.add_argument('--eval', dest='eval', default=False,
     help='eval', action='store_true')
    parser.add_argument('--time', dest='time', type=int, default=-1,
     help='time')
    parser.add_argument('--frame', dest='frame', type=str, default='',
     help='frame for time')
    parser.add_argument('--obj', dest='obj', type=str, default='',
     help='obj for time')
    return parser

def exec_bin(start, end):
    cmd = '%s -t %s -dir %s -start %s -end %s -config %s -seg-config %s -result-file %s'%(EVAL, TRACKLET, DIR, start, end, CONFIG, SEG_CONFIG, RESULT_FILE)
    print(cmd)
    subprocess.run(cmd, shell=True)

def exec_python(start, end, tech, time='-1'):
    cloud1 = DIR + start + "/" + end + ".txt"
    cloud2 = DIR + str(int(start) + 1) + "/groundless.txt"
    out = DIR + start + "/transform_" + end + "_" +  tech + ".txt"
    timeout = DIR + start + "/time_" + end + "_" +  tech + ".txt"
    cmd = 'python estimate_transform_rigid.py --cloud1 {} --cloud2 {} --out {} --mode {} --time {} --timeout {}'.format(cloud1, cloud2, out, tech, time, timeout)
    print(cmd)
    subprocess.run(cmd, shell=True)

def exec_svr(start, end):
    cloud1 = DIR + start + "/" + end + ".txt"
    cloud2 = DIR + str(int(start) + 1) + "/groundless.txt"
    config = "../svr/config.txt"
    out_dummy = "./svr_transformed.txt"
    tech = "svr"
    out = DIR + start + "/transform_" + end + "_" +  tech + ".txt"
    out_time = DIR + start + "/time_" + end + "_" +  tech + ".txt"
    cmd = '/home/gustavo/svr/distribution/C++/build/svr {} {} {} {} {} {}'.format(cloud2, cloud1, config, out_dummy, out, out_time)
    print(cmd)
    subprocess.run(cmd, shell=True)

def mat_to_pose(mat):
    angle_x = math.atan2(mat[2, 1], mat[2, 2])
    angle_y = math.atan2(-mat[2, 0], math.sqrt(mat[2, 1] * mat[2, 1] + mat[2, 2] * mat[2, 2]))
    angle_z = math.atan2(mat[1, 0], mat[0, 0])

    translation_x = mat[0, 3]
    translation_y = mat[1, 3]
    translation_z = mat[2, 3]
    return np.array([angle_x, angle_y, angle_z, translation_x, translation_y, translation_z])


def get_dist(tr_1, tr_2):
    translation_dist = tr_2[:3] - tr_1[:3]
    translation_dist = math.sqrt(np.dot(translation_dist, translation_dist))
    
    angle_dist = tr_2[3:] - tr_1[3:]
    angle_dist = math.sqrt(np.dot(angle_dist, angle_dist))
    
    return translation_dist, angle_dist


def evaluate_obj(start, end, tech):
    path_start = os.path.join(DIR, start, "transform_" + end + ".txt")
    path_icp = os.path.join(DIR, start, "transform_" + end + "_" + tech + ".txt")
    next_frame = int(start) + 1
    path_end = os.path.join(DIR, str(next_frame), "transform_" + end + ".txt")
    
    
    start_base = np.loadtxt(path_start)
    try:
        start_icp = np.loadtxt(path_icp)
    except FileNotFoundError as e:
        start_icp = np.identity(4)

    try:
        end_base = np.loadtxt(path_end)
    except FileNotFoundError as e:
        return

    estimated = np.dot(start_icp, start_base)
    r_estimated = mat_to_pose(estimated)

    r_gt = mat_to_pose(end_base)

    t_d, r_d = get_dist(r_estimated, r_gt)

    
    with open(METRIC_DIR + tech + '.csv', "a") as myfile:
        myfile.write("{}, {}, {}, {}\n".format(start, end, t_d, r_d))


from probreg.transformation import RigidTransformation
import open3d as o3
def get_trans(mat):
    trans_param = RigidTransformation()
    trans_param.rot = mat[:3, :3]
    trans_param.t = mat[:3, -1]
    return trans_param

def evaluate_obj2(start, end, tech):
    path_start = os.path.join(DIR, start, "transform_" + end + ".txt")
    path_icp = os.path.join(DIR, start, "transform_" + end + "_" + tech + ".txt")
    next_frame = int(start) + 1
    path_end = os.path.join(DIR, str(next_frame), "transform_" + end + ".txt")
    
    cloud_path = os.path.join(DIR, start, end + ".txt")
    cloud = o3.io.read_point_cloud(cloud_path, format='xyz')

    start_base = np.loadtxt(path_start)
    try:
        start_icp = np.loadtxt(path_icp)
    except IOError as e:
        start_icp = np.identity(4)

    try:
        end_base = np.loadtxt(path_end)
    except IOError as e:
        return

    inv_base = np.linalg.inv(start_base)
    
    original_points = get_trans(inv_base).transform(cloud.points)
    
    estimated = np.dot(start_icp, start_base)
    
    points = get_trans(estimated).transform(original_points)
    
    points_gt = get_trans(end_base).transform(original_points)
    
    diff = np.asarray(points_gt) - np.asarray(points)
    avg_dist = np.mean(np.sqrt(np.sum(diff*diff, axis=1)))

    with open(METRIC_DIR + tech + '.csv', "a") as myfile:
        myfile.write("{}, {}, {}\n".format(start, end, avg_dist))


def evaluate(frames, tech):
    #for frame in frames[:100]:
    for frame in [frames[117], frames[233]]:     
        frame_path = os.path.join(DIR, frame)
        for subdir, dirs, files in os.walk(os.path.abspath(frame_path)):
            for file in files:
                names = file.split('.')
                if names[0].isdigit():
                    evaluate_obj2(frame, names[0], tech)

def evaluate_time(times, frame, obj, tech):
    exec_function = exec_bin
    if tech == "svr":
        exec_function = exec_svr
    if tech == "filterreg" or tech == 'gmmtree':
        exec_function = lambda f, o: exec_python(f, o, tech, time='1')
    for i in range(times):
        exec_function(frame, obj)

if __name__=="__main__":
    args = init_parser().parse_args()
    frames = []
    for subdir, dirs, files in os.walk(os.path.abspath(DIR)):
        for frame in dirs:
            frames.append(int(frame))
    frames.sort()
    frames = list(map(str, frames))
    exec_function = exec_svr
    if args.eval:
        evaluate(frames, args.mode)
        exit()
    if args.time >= 0:
        evaluate_time(args.time, args.frame, args.obj, args.mode)
        exit()
    if args.mode == "c2ICP" or args.mode == "pmICP" or args.mode == "trICP" or args.mode == "SHOT_RANSAC":
        exec_function = exec_bin
    if args.mode == "gmmtree" or args.mode == "filterreg":
        exec_function = lambda f, n: exec_python(f, n, args.mode)
    for frame in frames:       
    #for frame in [frames[117], frames[233]]:
        frame_path = os.path.join(DIR, frame)
        for subdir, dirs, files in os.walk(os.path.abspath(frame_path)):
            for file in files:
                names = file.split('.')
                if names[0].isdigit():
                    print(names[0], frame)
                    exec_function(frame, names[0])
                
