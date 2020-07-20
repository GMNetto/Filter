#pragma once

#include <unordered_map>
#include <pcl/common/common_headers.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>

template <typename Point>
void estimate_normal(typename pcl::PointCloud<Point>::Ptr cloud, float radius) {
    typename pcl::NormalEstimation<Point, Point> ne;
    ne.setInputCloud (cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    typename pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point> ());
    ne.setSearchMethod (tree);

    // Output datasets
    typename pcl::PointCloud<Point>::Ptr cloud_normals (new pcl::PointCloud<Point>);

    // Use all neighbors in a sphere of radius 3cm
    //ne.setRadiusSearch (radius);

    ne.setKSearch(25);

    // Compute the features
    ne.compute (*cloud);
}