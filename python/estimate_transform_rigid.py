import os
import argparse
import subprocess
import re
import timeit
from probreg import filterreg, gmmtree
from probreg import features

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


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = 0.4
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3.registration.compute_fpfh_feature(
        pcd_down,
        o3.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(path_start, path_end, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3.io.read_point_cloud(path_start, format='xyz')
    target = o3.io.read_point_cloud(path_end, format='xyz')
    
    cloud_model = np.asarray(source.points)
    cloud_target = np.asarray(target.points)
    
    tree = o3.geometry.KDTreeFlann()
    tree.set_geometry(target)
    
    n_idxs = o3.utility.IntVector()
    for i in range(cloud_model.shape[0]):
        ns = tree.search_radius_vector_3d(cloud_model[i, :], 1.2)[1]
        n_idxs.extend(ns)
    n_idxs = np.asarray(n_idxs)
    n_idxs = np.unique(n_idxs, axis=0)
    sampled_target = cloud_target[n_idxs]
    
    target.points = o3.utility.Vector3dVector(sampled_target)
#     trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
#                              [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
#     source.transform(trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = 1.2
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3.registration.TransformationEstimationPointToPoint(False), 30, [
            o3.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3.registration.RANSACConvergenceCriteria(4000000, 50))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, transformation):
    distance_threshold = 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3.registration.TransformationEstimationPointToPoint())
    return result

def estimate_mat_open3d(path_start, path_end, tech="open3d"):
    voxel_size = 0.005 # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(path_start, path_end, voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)

    result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
                                 voxel_size, result_ransac.transformation)
    
    mat = np.identity(4)
    mat[:3,:3] = result_icp.transformation[:3,:3]
    mat[:-1, -1] = result_icp.transformation[:-1,-1]
    
    return mat, None


def estimate_mat(path_start, path_end, tech="filterreg"):
    f1_object = o3.io.read_point_cloud(path_start, format='xyz')
    f2_object = o3.io.read_point_cloud(path_end, format='xyz')

    # f2_object.estimate_normals(search_param=o3.geometry.KDTreeSearchParamHybrid(
    #         radius=0.1, max_nn=30))
    # f1_object.estimate_normals(search_param=o3.geometry.KDTreeSearchParamHybrid(
    #         radius=0.1, max_nn=30))
    
    if tech == "filterreg":
        objective_type = 'pt2pt'

        #feat = features.FPFH(radius_normal=0.15, radius_feature=0.15)

        center = np.average(f1_object.points, axis=0)
        tree = o3.geometry.KDTreeFlann()
        tree.set_geometry(f2_object)
        vectors = []
        res_idxs = tree.search_radius_vector_3d(center, 1.2)[1]
        new_groundless = np.asarray(f2_object.points)[res_idxs,:]
        print(new_groundless.shape)

        #tf_param, _, _ = filterreg.registration_filterreg(f1_object, new_groundless, sigma2=0.03, maxiter=10, feature_fn=feat, tol=0.001)
        tf_param, _, _ = filterreg.registration_filterreg(f1_object, f2_object, sigma2=0.03, maxiter=10, tol=0.0001)
    else:
        tf_param, a = gmmtree.registration_gmmtree(f1_object, f2_object, maxiter=20, tol=1.0e-3, lambda_c=1.0e-3)

    # res = tf_param.transform(f1_object.points)
    # test2 = o3.geometry.PointCloud()
    # test2.points = o3.utility.Vector3dVector(res)

    # o3.visualization.draw_geometries([test2,f2_object])

    mat = np.identity(4)
    mat[:3,:3] = tf_param.rot
    mat[:-1, -1] = tf_param.t
    return mat, tf_param

if __name__=="__main__":
    args = init_parser().parse_args()

    if int(args.time) >= 0:
        stmt = ''
        if args.mode == "filterreg":
            stmt = 'estimate_mat(args.cloud1, args.cloud2)'
        if args.mode == "gmmtree":
            stmt = 'estimate_mat(args.cloud1, args.cloud2, "gmmtree")'
        if args.mode == "open3d":
            stmt = 'estimate_mat_open3d(args.cloud1, args.cloud2, "gmmtree")'
        ex_time = timeit.timeit(stmt=stmt, number = 1, setup="from __main__ import estimate_mat, estimate_mat_open3d, args")
        int_time = int(ex_time*1000)
        with open(args.timeout, 'a') as f:
            f.write(str(int_time) + '\n')
        print('exec time:', ex_time)
        exit()
    if args.mode == "filterreg":
        trans_fR, trans_param = estimate_mat(args.cloud1, args.cloud2)
    if args.mode == "gmmtree":
        trans_fR, trans_param = estimate_mat(args.cloud1, args.cloud2, "gmmtree")
    if args.mode == "open3d":
        trans_fR, trans_param = estimate_mat_open3d(args.cloud1, args.cloud2, "open3d")
    np.savetxt(args.out, trans_fR, fmt='%.4f')
    
