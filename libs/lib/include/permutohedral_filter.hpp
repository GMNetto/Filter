#pragma once

#include <pcl/common/common_headers.h>
#include <pcl/ml/permutohedral.h>

#include "util.hpp"

class PermutohedralFilter
{
    private:
    float sigma_s=2, sigma_r=0.055;
    
    public:
    PermutohedralFilter(InputParams &input_params) {
        sigma_s = input_params.sigma_s;
        sigma_r = input_params.sigma_r;
    }

    void filter(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud
    , pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs
    , pcl::PointCloud<pcl::PointXYZINormal>::Ptr result);
};


