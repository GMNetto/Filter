import os
import argparse
import subprocess
import re
import math
import numpy as np

PROJECT_PATH = '/home/gustavo/filter/'

CONFIG = PROJECT_PATH + "/configs/rigid/config_seq91_teaser.json"
SEG_CONFIG = PROJECT_PATH + '/configs/config_seg.json'
TRACKLET = PROJECT_PATH + '/segmented/91/tracklet_labels.xml'
DIR = PROJECT_PATH + "/segmented/91/"
RESULT_FILE = PROJECT_PATH + "/segmented/91/"
EVAL = PROJECT_PATH + "/build/apps/match/match -match"
METRIC_DIR = './'

FLOWNET='/home/gustavo/filter/build/apps/app/app -flownet2'
TRANSFORMED_OBJ_PATH='/home/gustavo/filter/flownet/ex_trans.txt'
RES_DIR='/home/gustavo/filter/segmented_npz/91_full_results'

FILTERREG_EXTERNAL = '/home/gustavo/filterreg_objects/'

def evaluate_filterreg_external(frame, name, tech):
    path_start = os.path.join(DIR, frame, "transform_" + name + ".txt")
    next_frame = int(frame) + 1
    path_end = os.path.join(DIR, str(next_frame), "transform_" + name + ".txt")
    
    cloud_path = os.path.join(DIR, frame, name + ".txt")
    cloud = o3.io.read_point_cloud(cloud_path, format='xyz')

    start_base = np.loadtxt(path_start)

    try:
        end_base = np.loadtxt(path_end)
    except IOError as e:
        return

    inv_base = np.linalg.inv(start_base)
    
    original_points = get_trans(inv_base).transform(cloud.points)
        
    res_name = FILTERREG_EXTERNAL+frame+'_'+name+'.txt'
    print(res_name)
    points = o3.io.read_point_cloud(res_name, format='xyz').points
    
    points_gt = get_trans(end_base).transform(original_points)
    

    diff = np.asarray(points_gt) - np.asarray(points)
    avg_dist = np.mean(np.sqrt(np.sum(diff*diff, axis=1)))

    print('RESULT', avg_dist)
    with open(METRIC_DIR + 'EXTERNAL_FILTERREG.csv', "a") as myfile:
        myfile.write("{}, {}, {}\n".format(frame, name, avg_dist))




#until 300
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

def exec_go_icp(start, end, tech, time='-1'):
    cloud1 = DIR + start + "/" + end + ".txt"
    cloud2 = DIR + str(int(start) + 1) + "/groundless.txt"
    out = DIR + start + "/transform_" + end + "_" +  tech + ".txt"
    timeout = DIR + start + "/time_" + end + "_" +  tech + ".txt"
    cmd = '/usr/bin/python3 estimate_transform_go_icp.py --cloud1 {} --cloud2 {} --out {} --mode {} --time {} --timeout {}'.format(cloud1, cloud2, out, tech, time, timeout)
    print(cmd)
    subprocess.run(cmd, shell=True)

def exec_python(start, end, tech, time='-1'):
    cloud1 = DIR + start + "/" + end + ".txt"
    cloud2 = DIR + str(int(start) + 1) + "/groundless.txt"
    out = DIR + start + "/transform_" + end + "_" +  tech + ".txt"
    timeout = DIR + start + "/time_" + end + "_" +  tech + ".txt"
    cmd = '/usr/bin/python3 estimate_transform_rigid.py --cloud1 {} --cloud2 {} --out {} --mode {} --time {} --timeout {}'.format(cloud1, cloud2, out, tech, time, timeout)
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


def evaluate_go_icp(start, end, tech):
    path_start = os.path.join(DIR, start, "transform_" + end + ".txt")
    path_icp = os.path.join(DIR, start, "transform_" + end + "_" + tech + ".txt")
    next_frame = int(start) + 1
    path_end = os.path.join(DIR, str(next_frame), "transform_" + end + ".txt")
    
    cloud_path = os.path.join(DIR, start, end + ".txt")
    cloud = o3.io.read_point_cloud(cloud_path, format='xyz')

    start_base = np.loadtxt(path_start)
    # try:
    #     start_icp = np.loadtxt(path_icp)
    # except IOError as e:
    #     start_icp = np.identity(4)

    try:
        end_base = np.loadtxt(path_end)
    except IOError as e:
        return

    inv_base = np.linalg.inv(start_base)
    
    original_points = get_trans(inv_base).transform(cloud.points)
    
    #estimated = np.dot(start_icp, start_base)
    
    points = np.loadtxt(path_icp) #get_trans(estimated).transform(original_points)
    
    points_gt = get_trans(end_base).transform(original_points)
    
    diff = np.asarray(points_gt) - np.asarray(points)
    avg_dist = np.mean(np.sqrt(np.sum(diff*diff, axis=1)))

    with open(METRIC_DIR + tech + '.csv', "a") as myfile:
        myfile.write("{}, {}, {}\n".format(start, end, avg_dist))


def eval_flownet_bin(object_path, cloud_res, cloud_pos1, result_file):
    cmd = f'{FLOWNET} -f1 {object_path} -f2 {cloud_res} -f3 {cloud_pos1} -config {CONFIG} -seg-config {SEG_CONFIG} -result-file {result_file}'
    print(cmd)
    subprocess.run(cmd, shell=True)

def create_tmp_res_files(res_file):
    data_c = np.load(res_file)
    pos1 = data_c['pos1'][0]
    f_pos1 = '/home/gustavo/filter/flownet/ex_pos1.txt'
    np.savetxt(f_pos1, pos1, delimiter=' ', fmt='%.4f')

    res =  data_c['res'][0]
    f_res = '/home/gustavo/filter/flownet/ex_res.txt'
    np.savetxt(f_res, res, delimiter=' ', fmt='%.4f')
    return f_res, f_pos1

def evaluate_obj_flownet(frame, name, res_file):
    path_start = os.path.join(DIR, frame, "transform_" + name + ".txt")
    next_frame = int(frame) + 1
    path_end = os.path.join(DIR, str(next_frame), "transform_" + name + ".txt")
    
    cloud_path = os.path.join(DIR, frame, name + ".txt")
    cloud = o3.io.read_point_cloud(cloud_path, format='xyz')

    # Should send to c++ path to object, path to gr
    res_path, pos1_path = create_tmp_res_files(res_file)
    eval_flownet_bin(cloud_path, res_path, pos1_path, TRANSFORMED_OBJ_PATH)

    selected_cloud = o3.io.read_point_cloud(TRANSFORMED_OBJ_PATH+"2", format='xyz')
    start_base = np.loadtxt(path_start)

    try:
        end_base = np.loadtxt(path_end)
    except IOError as e:
        return

    inv_base = np.linalg.inv(start_base)
    
    original_points = get_trans(inv_base).transform(selected_cloud.points)
        
    points = o3.io.read_point_cloud(TRANSFORMED_OBJ_PATH, format='xyz').points
    
    points_gt = get_trans(end_base).transform(original_points)

    np.savetxt('/home/gustavo/filter/flownet/dummy.txt', np.asarray(points_gt), delimiter=' ', fmt='%.4f')
    
    diff = np.asarray(points_gt) - np.asarray(points)
    avg_dist = np.mean(np.sqrt(np.sum(diff*diff, axis=1)))

    print('RESULT', avg_dist)
    with open(METRIC_DIR + 'FLOWNET.csv', "a") as myfile:
        myfile.write("{}, {}, {}\n".format(frame, name, avg_dist))

def evaluate_flownet(frames):
    res_files = []
    for subdir, dirs, files in os.walk(RES_DIR):
        res_files = files
    res_files = sorted(res_files, key = lambda x: int(x[5:-4]))
    res_files = [os.path.join(RES_DIR, f) for f in res_files]
    for i, frame in enumerate(frames[3:]):
        i += 3
        frame_path = os.path.join(DIR, frame)
        for subdir, dirs, files in os.walk(os.path.abspath(frame_path)):
            for file in files:
                names = file.split('.')
                if names[0].isdigit():
                    print('res file', res_files[i])
                    evaluate_obj_flownet(frame, names[0], res_files[i])
                #break

def evaluate(frames, tech):
    print('TECH: ', tech)
    if tech=="FLOWNET":
        evaluate_flownet(frames)
        return
    #for frame in frames[:100]:
    for frame in frames[4:100]:
    #for frame in [frames[117], frames[233]]:     
        frame_path = os.path.join(DIR, frame)
        for subdir, dirs, files in os.walk(os.path.abspath(frame_path)):
            for file in files:
                names = file.split('.')
                if names[0].isdigit():
                    if tech=="EXTERNAL_FILTERREG":
                        evaluate_filterreg_external(frame, names[0], tech)
                    if tech=="go-icp":
                        evaluate_go_icp(frame, names[0], tech)
                    else:
                        evaluate_obj2(frame, names[0], tech)

def evaluate_time(times, frame, obj, tech):
    exec_function = exec_bin
    if tech == "svr":
        exec_function = exec_svr
    if tech == "filterreg" or tech == 'gmmtree' or tech == 'open3d':
        exec_function = lambda f, o: exec_python(f, o, tech, time='1')
    if tech == "go-icp":
        exec_function = lambda f, n: exec_go_icp(f, n, args.mode)
    print("TIME")
    for i in range(times):
        exec_function(frame, obj)
        evaluate_obj2(frame, obj, tech)

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
    if args.mode == "c2ICP" or args.mode == "pmICP" or args.mode == "trICP" or args.mode == "SHOT_RANSAC" or args.mode == "TEASER_FEATURE" or args.mode == "TEASER_FEATURE_POINT":
        exec_function = exec_bin
    if args.mode == "gmmtree" or args.mode == "filterreg" or args.mode == "open3d":
        exec_function = lambda f, n: exec_python(f, n, args.mode)
    if args.mode == "go-icp":
        exec_function = lambda f, n: exec_go_icp(f, n, args.mode)
    for frame in frames[4:100]:       
    #for frame in [frames[117], frames[233]]:
        frame_path = os.path.join(DIR, frame)
        for subdir, dirs, files in os.walk(os.path.abspath(frame_path)):
            for file in files:
                names = file.split('.')
                if names[0].isdigit():
                    print(names[0], frame)
                    exec_function(frame, names[0])
                
