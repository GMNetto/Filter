#pragma once

#include <unordered_map>
#include <pcl/common/common_headers.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/keypoints/keypoint.h>

#include "match.hpp"
#include "util.hpp"

class Keypoint
{
    private:
    std::string harris_method, tech;
    float harris_radius = 1, harris_threshold = 1, uniform_radius = 1.1;
    
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree;

    std::shared_ptr<std::vector<int>> get_harris_keypoints
    (pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr input);

    std::shared_ptr<std::vector<int>> get_uniform_keypoints
    (pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr input);

    std::shared_ptr<std::vector<int>>
    get_image_keypoints
    (pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr input,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr surface);

    std::shared_ptr<std::vector<int>>
    get_multiscale_keypoints(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr input);  

    public:
    //Make default equals to doc
    Keypoint(std::string tech="HARRIS", std::string harris_method="TOMASI",
             float harris_radius=1, float harris_threshold=1,
             float uniform_radius=1.1) {
                
        this->harris_method = harris_method;
        this->tech = tech;
        this->harris_radius = harris_radius;
        this->harris_threshold = harris_threshold;

        this->uniform_radius = uniform_radius;
    }

    Keypoint(InputParams &input_params) {
        this->harris_method = input_params.keypoint_method;
        this->tech = input_params.tech;
        
        this->uniform_radius = input_params.uniform_radius;
        this->harris_radius = input_params.keypoint_radius; 
        this->harris_threshold = input_params.keypoint_threshold;

        tree = boost::make_shared<pcl::search::KdTree<pcl::PointXYZRGBNormal>>();
    }

    std::shared_ptr<std::vector<int>>
    get_keypoints
    (pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr input, pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr surface=NULL);  
};

void get_descriptors(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr input,
                     std::shared_ptr<std::vector<int>> keypoints,
                     pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input_keypoints,
                     pcl::PointCloud<pcl::SHOT352>::Ptr input_descriptors, float radius);

template <typename Target, typename Source>
void assign_keypoints(typename pcl::PointCloud<Target>::Ptr cloud, typename std::unordered_map<int, Source>& vectors) {
    for (int i=0; i < cloud->size(); i++) {
        cloud->points[i].normal_x = 0;
        cloud->points[i].normal_y = 0;
        cloud->points[i].normal_z = 0;
        cloud->points[i].intensity = 0;
    }
    for (auto& it: vectors) {
        Source& c_vector =  it.second; 
        //std::cout << "Assign " << it.first << " " << c_vector << std::endl;
        cloud->points[it.first].normal_x = c_vector.x;
        cloud->points[it.first].normal_y = c_vector.y;
        cloud->points[it.first].normal_z = c_vector.z;
        cloud->points[it.first].intensity = 1;
    }

};
