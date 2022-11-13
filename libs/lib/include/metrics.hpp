#pragma once

#include <stdio.h>
#include <cmath>
#include <pcl/common/common_headers.h>
#include <pcl/search/kdtree.h>

#include "util.hpp"


template <typename Point>
void compute_hausdorff (typename pcl::PointCloud<Point>::Ptr cloud_a, typename pcl::PointCloud<Point>::Ptr cloud_b)
{

  // compare A to B
  typename pcl::search::KdTree<Point> tree_b;
  tree_b.setInputCloud (cloud_b);
  float max_dist_a = -std::numeric_limits<float>::max ();
  for (size_t i = 0; i < cloud_a->points.size (); ++i)
  {
    std::vector<int> indices (1);
    std::vector<float> sqr_distances (1);

    tree_b.nearestKSearch (cloud_a->points[i], 1, indices, sqr_distances);
    if (sqr_distances[0] > max_dist_a)
      max_dist_a = sqr_distances[0];
  }

  // compare B to A
  typename pcl::search::KdTree<Point> tree_a;
  tree_a.setInputCloud (cloud_a);
  float max_dist_b = -std::numeric_limits<float>::max ();
  for (size_t i = 0; i < cloud_b->points.size (); ++i)
  {
    std::vector<int> indices (1);
    std::vector<float> sqr_distances (1);

    tree_a.nearestKSearch (cloud_b->points[i], 1, indices, sqr_distances);
    if (sqr_distances[0] > max_dist_b)
      max_dist_b = sqr_distances[0];
  }

  max_dist_a = std::sqrt (max_dist_a);
  max_dist_b = std::sqrt (max_dist_b);

  float dist = std::max (max_dist_a, max_dist_b);

  std::cout << "A->B: " << " " << max_dist_a << std::endl;
  std::cout << "B->A: " << " " << max_dist_b << std::endl;
  std::cout << "Hausdorff Distance: " << " " << dist << std::endl;
}

//Average angle metric
template <typename Normal1, typename Normal2>
float average_angle(typename pcl::PointCloud<Normal1>::Ptr cloud_a, typename pcl::PointCloud<Normal2>::Ptr cloud_b) {
  float summed_angle = 0;
  for (int i = 0; i < cloud_a->size(); i++) {

    Normal1 pn = cloud_a->points[i];
    Normal2 nn = cloud_b->points[i];

    float cos_val = dot_product_normal<Normal1, Normal2>(pn, nn);
    summed_angle += fabs(acos(std::min(1.0f, cos_val)));
  }
  return summed_angle/cloud_a->size();
}

//Average distance metric
template <typename Point>
float average_point_distance(typename pcl::PointCloud<Point>::Ptr cloud_a, typename pcl::PointCloud<Point>::Ptr cloud_b) {
  float summed_distance = 0;
  for (int i = 0; i < cloud_a->size(); i++) {

    Point p = cloud_a->points[i], n = cloud_b->points[i];

    Point diff;
    diff.getArray3fMap() = p.getArray3fMap() - n.getArray3fMap();
    if (i == 1839) {
      std::cout << p << std::endl;
      std::cout << n << std::endl;
    }
  
    summed_distance += norm<Point>(diff);
    //std::cout << i << " " << summed_distance << " " << diff << std::endl;
  }
  return summed_distance/cloud_a->size();
}

void crispness_clouds(std::string f1, std::string f2, std::string result_file, InputParams &input_params);
void crispness_clouds_2(std::string f1, std::string f2, std::string result_file, InputParams &input_params);

pcl::PointXYZ crispness(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud,
 pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
 std::shared_ptr<std::vector<int>> points=NULL);

float crispness(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
 pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
 std::shared_ptr<std::vector<int>> points=NULL);

float crispness3(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
 pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2);

float sum_of_distance_GT(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud);

float average_of_distance_GT(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud);