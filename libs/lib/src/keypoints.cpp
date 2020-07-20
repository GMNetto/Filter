#include "keypoints.hpp"

#include "util.hpp"

#include <pcl/octree/octree_search.h>
#include <pcl/common/common_headers.h>
#include <pcl/search/kdtree.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/features/shot_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/geometry.h>
#include <pcl/registration/default_convergence_criteria.h>
#include <pcl/features/multiscale_feature_persistence.h>
#include <pcl/features/fpfh.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/filters/uniform_sampling.h>

#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <algorithm>
#include <math.h>

std::shared_ptr<std::vector<int>> Keypoint::get_uniform_keypoints
    (pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr input)
{
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBNormal> ());
    tree->setInputCloud(input);

    pcl::UniformSampling<pcl::PointXYZRGBNormal> uniform_detector;
    uniform_detector.setInputCloud(input);
    uniform_detector.setRadiusSearch(this->uniform_radius);

    std::cout << "Uniform radius " << this->uniform_radius << std::endl;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    uniform_detector.filter (*temp);

    std::cout << "Number samples " << temp->size() << " " << input->size() << std::endl;

    std::vector<int> nn_indices;
    std::vector<float> nn_dists;
    std::shared_ptr<std::vector<int>> result = std::make_shared<std::vector<int>>();
    result->reserve(temp->size());
    std::cout << "Number samples " << result->size() << std::endl;
    for (int i = 0; i < temp->size(); i++) {
        pcl::PointXYZRGBNormal &p = temp->points[i];
        if (isnan(p.x)) continue;
        tree->nearestKSearch(p, 1, nn_indices, nn_dists);
        result->push_back(nn_indices[0]);
    }
    std::cout << "Number samples " << result->size() << std::endl;
    return result;
    //return NULL;
}

std::shared_ptr<std::vector<int>>
Keypoint::get_harris_keypoints
(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr input)
{
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBNormal> ());

    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints (new pcl::PointCloud<pcl::PointXYZI>);

    pcl::PointCloud<pcl::PointXYZI>::Ptr key_harris (new pcl::PointCloud<pcl::PointXYZI>);
    
    tree->setInputCloud (input);

    pcl::HarrisKeypoint3D<pcl::PointXYZRGBNormal, pcl::PointXYZI, pcl::PointXYZRGBNormal>::ResponseMethod method;
    if (this->harris_method == "HARRIS") 
        method = pcl::HarrisKeypoint3D<pcl::PointXYZRGBNormal, pcl::PointXYZI, pcl::PointXYZRGBNormal>::HARRIS;
    else if(this->harris_method == "TOMASI")
        method = pcl::HarrisKeypoint3D<pcl::PointXYZRGBNormal, pcl::PointXYZI, pcl::PointXYZRGBNormal>::TOMASI;
    else if(this->harris_method == "CURVATURE")
        method = pcl::HarrisKeypoint3D<pcl::PointXYZRGBNormal, pcl::PointXYZI, pcl::PointXYZRGBNormal>::CURVATURE;

    
    pcl::HarrisKeypoint3D<pcl::PointXYZRGBNormal, pcl::PointXYZI, pcl::PointXYZRGBNormal> harris_detector(method, this->harris_radius, this->harris_threshold);
    harris_detector.setSearchMethod(tree);
    harris_detector.setNonMaxSupression (true);
    harris_detector.setRefine(false);
    harris_detector.setInputCloud (input);
    harris_detector.compute (*keypoints);

    pcl::PointIndicesConstPtr keypoints_indices = harris_detector.getKeypointsIndices ();
    return std::make_shared<std::vector<int>>(keypoints_indices->indices);
}

std::shared_ptr<std::vector<int>>
Keypoint::get_image_keypoints
(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr input, pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr surface)
{
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBNormal> ());

    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints (new pcl::PointCloud<pcl::PointXYZI>);

    pcl::PointCloud<pcl::PointXYZI>::Ptr key_harris (new pcl::PointCloud<pcl::PointXYZI>);
    
    if (surface)
        tree->setInputCloud (surface);

    pcl::HarrisKeypoint3D<pcl::PointXYZRGBNormal, pcl::PointXYZI, pcl::PointXYZRGBNormal>::ResponseMethod method;
    if (this->harris_method == "HARRIS") 
        method = pcl::HarrisKeypoint3D<pcl::PointXYZRGBNormal, pcl::PointXYZI, pcl::PointXYZRGBNormal>::HARRIS;
    else if(this->harris_method == "TOMASI")
        method = pcl::HarrisKeypoint3D<pcl::PointXYZRGBNormal, pcl::PointXYZI, pcl::PointXYZRGBNormal>::TOMASI;
    else if(this->harris_method == "CURVATURE")
        method = pcl::HarrisKeypoint3D<pcl::PointXYZRGBNormal, pcl::PointXYZI, pcl::PointXYZRGBNormal>::CURVATURE;

    
    pcl::HarrisKeypoint3D<pcl::PointXYZRGBNormal, pcl::PointXYZI, pcl::PointXYZRGBNormal> harris_detector(method, this->harris_radius, this->harris_threshold);
    harris_detector.setSearchMethod(tree);
    harris_detector.setNonMaxSupression (true);
    harris_detector.setRefine(false);
    if (surface)
        harris_detector.setSearchSurface (surface);
    harris_detector.setInputCloud (input);
    harris_detector.compute (*keypoints);

    std::cout << "Input " << input->size() << " " << surface->size() << std::endl;

    pcl::PointIndicesConstPtr keypoints_indices = harris_detector.getKeypointsIndices ();
    return std::make_shared<std::vector<int>>(keypoints_indices->indices);
    //return std::make_shared<std::vector<int>>(input->indices);
}

std::shared_ptr<std::vector<int>>
Keypoint::get_multiscale_keypoints
(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr input) {
    pcl::FPFHEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::FPFHSignature33>::Ptr descr_est(new pcl::FPFHEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::FPFHSignature33>());
    //descr_est.setRadiusSearch (radius);
    descr_est->setInputCloud (input);
    descr_est->setInputNormals (input);
    //descr_est.setSearchSurface (input);
    //descr_est.compute (*input_descriptors);

    boost::shared_ptr<std::vector<int>> kps(new std::vector<int>());

    pcl::MultiscaleFeaturePersistence<pcl::PointXYZRGBNormal, pcl::FPFHSignature33> fper;
    std::vector<float> scale_values = { 1.0f };
    fper.setScalesVector (scale_values);
    fper.setAlpha (2.0f);
    fper.setFeatureEstimator (descr_est);
    fper.setDistanceMetric (pcl::CS);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features (new pcl::PointCloud<pcl::FPFHSignature33> ());
    fper.determinePersistentFeatures (*features, kps);
    return std::make_shared<std::vector<int>>(kps->begin(), kps->end());
}

std::shared_ptr<std::vector<int>>
Keypoint::get_keypoints(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr input, pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr surface) {
    if (this->tech == "HARRIS") {
        return this->get_harris_keypoints(input);
    } else if (this->tech == "UNIFORM") {
        return this->get_uniform_keypoints(input);
    }
    return this->get_multiscale_keypoints(input);
}

void get_descriptors(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr input,
                     std::shared_ptr<std::vector<int>> keypoints,
                     pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input_keypoints,
                     pcl::PointCloud<pcl::SHOT352>::Ptr input_descriptors, float radius) {
    pcl::copyPointCloud (*input, *keypoints, *input_keypoints);

    // Define descriptors
    pcl::SHOTEstimationOMP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::SHOT352> descr_est;
    std::cout << radius << std::endl;
    descr_est.setRadiusSearch (radius);

    descr_est.setInputCloud (input_keypoints);
    descr_est.setInputNormals (input);
    descr_est.setSearchSurface (input);
    descr_est.compute (*input_descriptors);
}


void match_keypoints(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr input1_keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr input2_keypoints,
                    pcl::PointCloud<pcl::SHOT352>::Ptr input1_descriptors,
                    pcl::PointCloud<pcl::SHOT352>::Ptr input2_descriptors,
                    std::vector<int>& prev2next) {

    // Use a KdTree to search for the nearest matches in feature space
    pcl::KdTreeFLANN<pcl::SHOT352> descriptor_kdtree;
    descriptor_kdtree.setInputCloud (input2_descriptors);

    // Find the index of the best match for each keypoint, and store it in "correspondences_out"
    const int k = 1;
    std::vector<int> k_indices (k);
    std::vector<float> k_squared_distances (k);
    for (int i = 0; i < static_cast<int> (input1_descriptors->size ()); ++i)
    {
        descriptor_kdtree.nearestKSearch (*input1_descriptors, i, k, k_indices, k_squared_distances);
        int idx = k_indices[0];

        float distance = l2_SHOT352(input1_descriptors->points[i],
                                input2_descriptors->points[idx]);


        pcl::PointXYZ diff;
        diff.getArray3fMap() = input2_keypoints->points[idx].getArray3fMap() - input1_keypoints->points[i].getArray3fMap();
        prev2next[i] = idx;
    }
}