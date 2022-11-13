#pragma once

#include <unordered_map>

#include <omp.h>

#include <pcl/common/common_headers.h>
#include <pcl/correspondence.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl-1.8/pcl/registration/registration.h>
#include <pcl-1.8/pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ml/permutohedral.h>

#include "util.hpp"

class PLMatchTwoWays
{
    private:
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, float> estimation;
    pcl::Permutohedral p;
    float sigma_s, start_sigma_s, sigma_r, patch_radius, match_radius;
    int patch_match_iter;
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> tree, tree2;

    void get_correspondences(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud2,
                    pcl::Correspondences &correspondences,
                    pcl::Correspondences &correspondences2);

    void filter(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs);

    float copyCorresp (pcl::Correspondences &correspondences, std::vector<int> &indices, pcl::Correspondences &tmp,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src);

   void transform_cloud(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud1,
                    pcl::Correspondences &correspondences,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud2,
                    pcl::Correspondences &correspondences2,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    void transform_keypoint(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src,
    int keypoint_idx,
    Eigen::Matrix4f &transformation,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    float get_local_correspondences(pcl::PointXYZRGBNormal &keypoint,
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> &tree,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
    pcl::Correspondences &correspondences,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr local_cloud,
    pcl::Correspondences &local_corresp);

    float get_transformation_cost(
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
    Eigen::Matrix4f &transformation,
    pcl::Correspondences &correspondences);

    public:

    PLMatchTwoWays(InputParams &input_params) {
        sigma_s = input_params.sigma_s;
        start_sigma_s = input_params.sigma_s;
        sigma_r = input_params.sigma_r;
        match_radius = input_params.match_radius;
        patch_radius = input_params.patch_radius;
        patch_match_iter = input_params.patchmatch_levels;
    }

    void local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);
};

class PLMatchSimple
{
    private:
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, float> estimation;
    pcl::Permutohedral p;
    float sigma_s, start_sigma_s, sigma_r, patch_radius, match_radius;
    int patch_match_iter;
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> tree, tree2;

    void get_new_pos(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud1,
                    std::vector<float> &weights);

    void filter(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs);

    void fill_valid_vectors(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_pos_cloud,
                    std::vector<float> &weights,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);
    
    public:

    PLMatchSimple(InputParams &input_params) {
        sigma_s = input_params.sigma_s;
        start_sigma_s = input_params.sigma_s;
        sigma_r = input_params.sigma_r;
        match_radius = input_params.match_radius;
        patch_radius = input_params.patch_radius;
        patch_match_iter = input_params.patchmatch_levels;
    }

    void local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);
};

class PLMatch
{
    private:
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, float> estimation;
    pcl::Permutohedral p;
    float sigma_s, start_sigma_s, sigma_r, patch_radius, match_radius;
    int patch_match_iter;
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> tree, tree2;

    void get_correspondences(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud1,
                    pcl::Correspondences &correspondences);

    void filter(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs);

    void copyCorresp (pcl::Correspondences &correspondences, std::vector<int> &indices, pcl::Correspondences &tmp,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src);

    void transform_cloud(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::Correspondences &correspondences,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    void transform_keypoint(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src,
    int keypoint_idx,
    Eigen::Matrix4f &transformation,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    public:

    PLMatch(InputParams &input_params) {
        sigma_s = input_params.sigma_s;
        start_sigma_s = input_params.sigma_s;
        sigma_r = input_params.sigma_r;
        match_radius = input_params.match_radius;
        patch_radius = input_params.patch_radius;
        patch_match_iter = input_params.patchmatch_levels;
    }

    void local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);
};

class DenseMatcher
{
    private:
    float match_radius, patch_radius;
    int max_iter_ransac, n_correspondence;
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> tree, tree2, tree_;
    pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;


    public:
    DenseMatcher(InputParams &input_params) {
        this->match_radius = input_params.match_radius;
        pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointXYZRGBNormal,
         pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr
        cor(new pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>);
        
        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>::Ptr
        rej_sample(new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>);
        rej_sample->setMaximumIterations(input_params.ransac_max_iter);

        icp.setCorrespondenceEstimation(cor);
        icp.addCorrespondenceRejector(rej_sample);

        patch_radius = input_params.patch_radius;

        icp.setMaxCorrespondenceDistance(input_params.match_radius);
        icp.setEuclideanFitnessEpsilon(input_params.icp_transformation_eps);
        icp.setMaximumIterations(input_params.icp_max_iter);
        icp.setRANSACIterations(input_params.ransac_max_iter);
    }

    void local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);
};