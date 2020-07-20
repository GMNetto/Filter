#pragma once 

#include <pcl/common/common_headers.h>


void pf_3D_point(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result, int iter, bool back, float sigma_s=100, float sigma_r=0.1, float sigma_d=0.7);