import argparse
import os
import subprocess
import numpy as np
from shutil import copyfile
import open3d as o3

# Adjust this to the current path
PROJECT_PATH = '/home/gustavo/fitler/'

OBJ = "cyclist"
TRACK = "1"
S1 = "0.050000"
S2 = "0.100000"
S3 = "0.200000"
S4 = "0.300000"
#IC1 = "/home/gustavo/fitler/experiments/density/"+OBJ+"/"+TRACK+".txt"
#IC1 = "/home/gustavo/fitler/experiments/density/"+OBJ+"/sampled/"+TRACK+"_"+S3+".txt"
#IC2 = "/home/gustavo/fitler/experiments/density/"+OBJ+"/next/"+TRACK+".txt"
#IC2 = "/home/gustavo/fitler/experiments/density/"+OBJ+"/next/"+TRACK+"_"+S3+".txt"
#IC1 = "/mnt/c/Users/gusta/Desktop/testKitti/horse1.txt"
#IC2 = "/mnt/c/Users/gusta/Desktop/testKitti/horse3.txt"
#IC1 = "/home/gustavo/fitler/segmented/corner_case_91/joined/1_2.txt"
#IC2 = "/home/gustavo/fitler/segmented/corner_case_91/joined28/1_2.txt"
IC1 = PROJECT_PATH + "python/mesh_0000.txt"
IC2 = PROJECT_PATH + "python/mesh_0015.txt"
#RESULT_FILE = '/mnt/c/Users/gusta/Pictures/Results/Article2/problem_2_people/joined_gmmreg.pcd'
RESULT_FILE = '/home/gustavo/filter/dummy.pcd'

# GMMReg compiled binary http://code.google.com/p/gmmreg/
BINARY = '/home/gustavo/gmmreg/C++/build/gmmreg_demo'

INI = ' ./test.ini'
MODE = 'tps_l2'
#MODE = 'em_grbf'
cloud1 = './cloud1.txt'
cloud2 = './cloud2.txt'
transformed = './transformed_model.txt'

# Compiled custom metric
EVAL = PROJECT_PATH + "build/apps/app/app -crispness"
CONFIG = PROJECT_PATH + "config_seq91.json"
SEG_CONFIG = PROJECT_PATH + "configs/config_seg.json"


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', type=str, default='single',
     help='mode')
    parser.add_argument('--start', dest='start', type=str, default='', 
     help='cloud 1')
    parser.add_argument('--end', dest='end', type=str, default='',
     help='cloud 2')
    parser.add_argument('--dir', dest='dir', type=str, default='',
     help='result file')
    return parser

class KO:

    def __init__(self, root, frame, track_id):
        self.frame = frame
        self.track_id = track_id
        self.root = root

    def get_path(self):
        return os.path.abspath(os.path.join(self.root, self.frame, self.track_id))

    def get_transformed_path(self):
        return os.path.abspath(os.path.join(self.root, self.frame, "t_" + self.track_id))


def load_objs(path, root):
    objs = {}
    print os.path.abspath(path)
    for subdir, dirs, files in os.walk(os.path.join(root, path)):
        for file in files:
            if file[0] == 't' or file[0] == 'g':
                continue
            ko = KO(root, path, file)
            objs[file] = ko
    return objs

def copy_result(cur):
    copyfile(transformed, cur.get_transformed_path())

def evaluate(cur, next):
    result_file = os.path.join(os.path.abspath(cur.root), "GMMTEG")
    print result_file
    cmd = '%s -f1 %s -f2 %s -config %s -seg-config %s -result-file %s'%(EVAL, cur.get_transformed_path(), next.get_path(), CONFIG, SEG_CONFIG, result_file)
    subprocess.call(cmd, shell=True)

def evaluate_path(cur_path, next_path, root):
    result_file = os.path.join(os.path.abspath(root), "GMMTEG")
    print result_file
    cmd = '%s -f1 %s -f2 %s -config %s -seg-config %s -result-file %s'%(EVAL, cur_path, next_path, CONFIG, SEG_CONFIG, result_file)
    subprocess.call(cmd, shell=True)

def execute_reg_path(current, next, result):
    print "Executing for: ", current, next
    copyfile(current, cloud1)
    copyfile(next, cloud2)
    cmd = '%s %s %s'%(BINARY, INI, MODE)
    subprocess.call(cmd, shell=True)
    copyfile(transformed, result)
    a = np.loadtxt(result)
    return a

def execute_reg(cur, next):
    print "Executing for: ", cur.get_path(), next.get_path()
    copyfile(cur.get_path(), cloud1)
    copyfile(next.get_path(), cloud2)
    cmd = '%s %s %s'%(BINARY, INI, MODE)
    subprocess.call(cmd, shell=True)
    copy_result(cur)
    a = np.loadtxt(cur.get_transformed_path())
    return a

def compare_objs(cur_objs, next_objs, root, frame):
    results = []
    for cur_obj, obj in cur_objs.iteritems():
        if cur_obj in next_objs:
            transf = execute_reg(obj, next_objs[cur_obj])
            results.append(transf)
    result_path = os.path.join(root, str(frame-1), "t.txt")
    nex_grd = os.path.join(root, str(frame), "groundless.txt")
    np.savetxt(result_path, np.concatenate(results), fmt='%.4f')
    evaluate_path(result_path, nex_grd, root)

def compare_single(current, next, result):
    transf = execute_reg_path(current, next, result)
    np.savetxt(result, transf, fmt='%.4f')

    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(transf)
    o3.io.write_point_cloud(RESULT_FILE, pcd)

    evaluate_path(result, next, './')
    source = o3.io.read_point_cloud(result, format='xyz')
    target = o3.io.read_point_cloud(next, format='xyz')
    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 0, 1])
    o3.visualization.draw_geometries([source, target])

# def evaluate_path(next_path):
#     cmd = '%s -f1 %s -f2 %s -config %s -seg-config %s -result-file %s'%(EVAL, RESULT_FILE, next_path, CONFIG, SEG_CONFIG, METRIC_FILE)
#     #subprocess.call(cmd, shell=True)
#     print(cmd)
#     p = subprocess.Popen(cmd, shell=True)
#     p.wait()

# def execute_reg(current_frame, next_frame):
#     cmd = '%s -vec -f1 %s -f2 %s -config %s -seg-config %s -result-file %s'%(BINARY, current_frame, next_frame, CONFIG, SEG_CONFIG, RESULT_FILE)
#     p = subprocess.Popen(cmd, shell=True)
#     p.wait()
# copyfile(current_frame, RESULT_FILE)


def compare_frames(current_frame, next_frame):
    transf = execute_reg_path(current_frame, next_frame, RESULT_FILE)
    #evaluate_path(next_frame)
    evaluate_path(RESULT_FILE, next_frame, './')

def evaluate_meshes(args):
    frames = []
    for subdir, dirs, files in os.walk(os.path.abspath(args.dir)):
        frames.extend(files)

    current_frame = frames[0]
    print(current_frame)
    current_frame_p = os.path.join(args.dir, current_frame)
    for frame in frames[1:2]:
        print(frame)
        next_frame = frame
        next_frame_p = os.path.join(args.dir, next_frame)
        compare_frames(current_frame_p, next_frame_p)
        current_frame = next_frame
        current_frame_p = os.path.join(args.dir, current_frame)

if __name__=="__main__":
    args = init_parser().parse_args()
    if args.mode == 'single':
        print "SINGLE"
        compare_single(IC1, IC2, 'result.txt')
        exit()
    if args.mode == 'mesh':
        evaluate_meshes(args)
        exit()
    dirs = os.walk(os.path.abspath(args.dir))
    frames = []
    for subdir, dirs, files in os.walk(os.path.abspath(args.dir)):
        for frame in dirs:
            frames.append(int(frame))
    frames.sort()
    current_frame = frames[0]
    current_objs = load_objs(str(current_frame), args.dir)
    for frame in frames[1:]:
        next_frame = frame
        next_objs = load_objs(str(next_frame), args.dir)
        compare_objs(current_objs, next_objs, args.dir, frame)
        current_objs = next_objs
        
