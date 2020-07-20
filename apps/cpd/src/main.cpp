#include <iostream>
#include <unordered_map>
#include <fstream>
#include <chrono>

// Tracking include
#include <pcl/tracking/particle_filter.h>
#include <pcl/tracking/particle_filter_omp.h>
#include <pcl/tracking/tracking.h>
#include <pcl/tracking/coherence.h>
#include <pcl/tracking/distance_coherence.h>
#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/log/core.hpp>

#include "visualize.hpp"
#include "util.hpp"
#include "normal.hpp"
#include "filterPF.hpp"
#include "match.hpp"
#include "icp.hpp"
#include "segment.hpp"
#include "kitti.hpp"

// --------------
// -----Help-----
// --------------
void
printUsage (const char* progName)
{
  std::cout << "\n\nUsage: "<<progName<<" [options]\n\n";
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
 
  boost::property_tree::ptree pt;
  boost::property_tree::read_json(config, pt);
  InputParams input_params(pt);
  std::cout << "READ CONFIG" << std::endl;

  std::string f1 = "",
      f2 = "",
      f3 = "";

  std::string dir = "/home/gustavo/DT3D/models/Block0.4/", result_file = "./results/";

  int start = 0, end = 4;

  if (pcl::console::find_argument (argc, argv, "-result-file") >= 0)
  {
    pcl::console::parse_argument (argc, argv, "-result-file", result_file);
  }
      
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
 
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  
  bool loaded = load_pair(f1, f2, cloud1, cloud2, input_params);
  
  pcl::tracking::KLDAdaptiveParticleFilterOMPTracker<pcl::PointXYZRGBNormal, pcl::tracking::ParticleXYZRPY> tracker(8);

  pcl::tracking::ParticleXYZRPY bin_size;
  bin_size.x = 0.1f;
  bin_size.y = 0.1f;
  bin_size.z = 0.1f;
  bin_size.roll = 0.1f;
  bin_size.pitch = 0.1f;
  bin_size.yaw = 0.1f;

  tracker.setMaximumParticleNum (10000);
  tracker.setDelta (0.99);
  tracker.setEpsilon (0.2);
  tracker.setBinSize (bin_size);
  
  // Covariances
  std::vector<double> default_step_covariance = std::vector<double> (6, 0.015 * 0.015);
  
  std::vector<double> initial_noise_covariance = std::vector<double> (6, 0.00001);
  std::vector<double> default_initial_mean = std::vector<double> (6, 0.0);
  
  tracker.setStepNoiseCovariance (default_step_covariance);
  tracker.setInitialNoiseCovariance (initial_noise_covariance);
  tracker.setInitialNoiseMean (default_initial_mean);
  tracker.setTrans (Eigen::Affine3f::Identity ());
  tracker.setIterationNum (10);
  tracker.setParticleNum (600);
  tracker.setResampleLikelihoodThr(0.00);
  tracker.setUseNormal (false);

  pcl::tracking::ApproxNearestPairPointCloudCoherence<pcl::PointXYZRGBNormal>::Ptr coherence
  (new pcl::tracking::ApproxNearestPairPointCloudCoherence<pcl::PointXYZRGBNormal>);

  pcl::tracking::DistanceCoherence<pcl::PointXYZRGBNormal>::Ptr distance_coherence
    (new pcl::tracking::DistanceCoherence<pcl::PointXYZRGBNormal>);
  coherence->addPointCoherence (distance_coherence);

  pcl::search::Octree<pcl::PointXYZRGBNormal>::Ptr search (new pcl::search::Octree<pcl::PointXYZRGBNormal> (0.01));
  coherence->setSearchMethod (search);
  coherence->setMaximumDistance (1.0);

  tracker.setCloudCoherence (coherence);
  Eigen::Vector4f c;
  Eigen::Affine3f trans = Eigen::Affine3f::Identity ();
  pcl::compute3DCentroid<pcl::PointXYZRGBNormal> (*cloud1, c);
  std::cout << "Centroid " << c << std::endl;
  trans.translation ().matrix () = Eigen::Vector3f (c[0], c[1], c[2]);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transed_ref (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::transformPointCloud<pcl::PointXYZRGBNormal> (*cloud1, *transed_ref, trans.inverse());
  
  // Sampled cloud2 for transformation estimation
  std::vector<int> points_cloud2;
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr s_cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  sample_cloud(cloud1, cloud2, points_cloud2, input_params.match_icp);
  pcl::copyPointCloud(*cloud2, points_cloud2, *s_cloud2);

  tracker.setReferenceCloud (transed_ref);
  tracker.setTrans(trans);
  tracker.setInputCloud(s_cloud2);
  tracker.compute();

  std::cout << "Particles " << tracker.getParticles()->points.size() << std::endl;
  
  pcl::tracking::ParticleXYZRPY result = tracker.getResult ();
  Eigen::Affine3f transformation = tracker.toEigenMatrix (result);

  std::cout << "estimated transformation " << transformation.matrix() << std::endl;

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::transformPointCloud<pcl::PointXYZRGBNormal> (*(tracker.getReferenceCloud ()), *transformed, transformation);
  
  for (int i = 0; i < cloud2->size(); i++) {
    cloud2->points[i].r = 40;
    cloud2->points[i].g = 40;
    cloud2->points[i].b = 40;
  }
  
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZINormal>);
  pcl::copyPointCloud(*transformed, *new_cloud);

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr merged (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  for (int i = 0; i < new_cloud->size(); i++) {
    new_cloud->points[i].normal_x = 0;
    new_cloud->points[i].normal_y = 0;
    new_cloud->points[i].normal_z = 0;
  }
  project_points(new_cloud, cloud2, merged);
  viewer = rgbVis(merged);

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }

  return 0;
}
