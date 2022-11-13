#include <iostream>
#include <unordered_map>
#include <fstream>
#include <chrono>
#include <random>

#include <pcl/common/common.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/ml/permutohedral.h>
#include <pcl/features/fpfh_omp.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "visualize.hpp"
#include "metrics.hpp"
#include "util.hpp"
#include "pf_vector.hpp"
#include "keypoints.hpp"
#include "normal.hpp"
#include "filterPF.hpp"
#include "match.hpp"
#include "icp.hpp"
#include "kitti.hpp"

// --------------
// -----Help-----
// --------------
void
printUsage (const char* progName)
{
  std::cout << "\n\nUsage: "<<progName<<" [options]\n\n";
}

std::shared_ptr<std::vector<int>> get_flownet_keypoints
    (pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr original,
     pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sampled)
{
  pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBNormal> ());
  tree->setInputCloud(original);

  std::vector<int> nn_indices;
  std::vector<float> nn_dists;
  std::shared_ptr<std::vector<int>> result = std::make_shared<std::vector<int>>();
  result->reserve(sampled->size());

  for (int i = 0; i < sampled->size(); i++) {
    pcl::PointXYZRGBNormal &p = sampled->points[i];
    if (isnan(p.x)) continue;
    tree->nearestKSearch(p, 1, nn_indices, nn_dists);
    //std::cout << "Dist: " << nn_dists[0] << std::endl;
    //if (nn_dists[0] < 0.01)
    result->push_back(nn_indices[0]);
  }

  return result;
}

int main(int argc, char **argv) {
  
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    printUsage (argv[0]);
    return 0;
  }

  std::string config = "config.json", seg_config = "";
  if (pcl::console::find_argument (argc, argv, "-config") >= 0) {
    pcl::console::parse_argument (argc, argv, "-config", config);
  }
  if (pcl::console::find_argument (argc, argv, "-seg-config") >= 0) {
      pcl::console::parse_argument (argc, argv, "-seg-config", seg_config);
  }
  boost::property_tree::ptree pt, pt_seg;
  boost::property_tree::read_json(config, pt);
  InputParams input_params(pt);
  std::cout << "READ CONFIG" << std::endl;

  boost::property_tree::read_json(seg_config, pt_seg);
  SegParams seg_params(pt_seg);
  std::cout << "READ SEG CONFIG" << std::endl;

  bool fpfh(false);
  std::string f1 = "", f2 = "", f3 = "", f4 = "", result_file = "";
  
  if (pcl::console::find_argument (argc, argv, "-result-file") >= 0)
  {
    pcl::console::parse_argument (argc, argv, "-result-file", result_file);
  }
      
  if (pcl::console::find_argument (argc, argv, "-fpfh") >= 0)
  {
    fpfh = true ;
    if (pcl::console::find_argument (argc, argv, "-f1") >= 0) {
      pcl::console::parse_argument (argc, argv, "-f1", f1);
    }
    std::cout << "Vec example " << fpfh << std::endl;
  }

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

  if (fpfh) {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr object_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr object (new pcl::PointCloud<pcl::PointNormal>);

    load_cloud(f1, object_cloud, input_params);
    pcl::copyPointCloud(*object_cloud, *object);

    
    pcl::FPFHEstimationOMP<pcl::PointNormal, pcl::PointNormal, pcl::FPFHSignature33> fest;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr object_features(new pcl::PointCloud<pcl::FPFHSignature33>());
    fest.setRadiusSearch(input_params.descriptor_radius);  
    fest.setInputCloud(object);
    fest.setInputNormals(object);
    fest.compute(*object_features);

    std::string m_str = "wb";
    const char *mode = m_str.c_str();
    FILE* fid = fopen(result_file.c_str(), mode);
    int nV = object->size(), nDim = 33;
    fwrite(&nV, sizeof(int), 1, fid);
    fwrite(&nDim, sizeof(int), 1, fid);
    for (int v = 0; v < nV; v++) {
        const pcl::PointNormal &pt = object->points[v];
        float xyz[3] = {pt.x, pt.y, pt.z};
        fwrite(xyz, sizeof(float), 3, fid);
        const pcl::FPFHSignature33 &feature = object_features->points[v];
        fwrite(feature.histogram, sizeof(float), 33, fid);
    }
    fclose(fid);
  }
  // Take three files as input output_x, output_y (calculate vectors)
  // Take output file, filter and 
  return 0;
}


