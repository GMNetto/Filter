import os
import argparse
import subprocess
import re
import timeit
from probreg import filterreg, gmmtree
import numpy as np
import open3d as o3

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud1', dest='cloud1', type=str, default='',
     help='cloud1')
    parser.add_argument('--cloud2', dest='cloud2', type=str, default='',
     help='cloud2')
    parser.add_argument('--out', dest='out', type=str, default='',
     help='out')
    parser.add_argument('--timeout', dest='timeout', type=str, default='',
     help='time out')
    parser.add_argument('--mode', dest='mode', type=str, default='',
     help='mode')
    parser.add_argument('--time', dest='time', type=int, default=-1,
     help='time')
    return parser

def estimate_mat(path_start, path_end, tech="filterreg"):
    f1_object = o3.io.read_point_cloud(path_start, format='xyz')
    f2_object = o3.io.read_point_cloud(path_end, format='xyz')

    f2_object.estimate_normals(search_param=o3.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
    f1_object.estimate_normals(search_param=o3.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
    
    if tech == "filterreg":
        objective_type = 'pt2pt'
        tf_param, _, _ = filterreg.registration_filterreg(f1_object, f2_object, f2_object.normals, objective_type=objective_type, sigma2=0.3, maxiter=10, tol=0.01)
    else:
        tf_param, a = gmmtree.registration_gmmtree(f1_object, f2_object, maxiter=20, tol=1.0e-3, lambda_c=1.0e-3)

    mat = np.identity(4)
    mat[:3,:3] = tf_param.rot
    mat[:-1, -1] = tf_param.t
    return mat, tf_param

if __name__=="__main__":
    args = init_parser().parse_args()

    if args.time >= 0:
        stmt = ''
        if args.mode == "filterreg":
            stmt = 'estimate_mat(args.cloud1, args.cloud2)'
        if args.mode == "gmmtree":
            stmt = 'estimate_mat(args.cloud1, args.cloud2, "gmmtree")'
        ex_time = timeit.timeit(stmt=stmt, number = 1, setup="from __main__ import estimate_mat, args")
        int_time = int(ex_time*1000)
        with open(args.timeout, 'a') as f:
            f.write(str(int_time) + '\n')
        print 'exec time:', ex_time
        exit()
    if args.mode == "filterreg":
        trans_fR, trans_param = estimate_mat(args.cloud1, args.cloud2)
    if args.mode == "gmmtree":
        trans_fR, trans_param = estimate_mat(args.cloud1, args.cloud2, "gmmtree")
    np.savetxt(args.out, trans_fR, fmt='%.4f')
    
