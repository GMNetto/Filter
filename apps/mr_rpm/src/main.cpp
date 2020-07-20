#include <iostream>
#include <unordered_map>
#include <fstream>
#include <chrono>

#include <pcl/common/common.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/conversions.h>
#include <pcl/surface/gp3.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/registration/correspondence_estimation.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/log/core.hpp>

#include "metrics.hpp"
#include "util.hpp"
#include "keypoints.hpp"
#include "normal.hpp"
#include "match.hpp"
#include "icp.hpp"

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

  std::string result_file = "./results/";

  if (pcl::console::find_argument (argc, argv, "-result-file") >= 0)
  {
    pcl::console::parse_argument (argc, argv, "-result-file", result_file);
  }

  if (pcl::console::find_argument (argc, argv, "-X") >= 0) {
    pcl::console::parse_argument (argc, argv, "-X", f1);
  }
  if (pcl::console::find_argument (argc, argv, "-Y") >= 0) {
    pcl::console::parse_argument (argc, argv, "-Y", f2);
  }
  if (pcl::console::find_argument (argc, argv, "-Z") >= 0) {
    pcl::console::parse_argument (argc, argv, "-Z", f3);
  }
  std::cout << "MR_RPM " << std::endl;

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud3 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  
  std::cout << "File 1 " << f1 << std::endl;
  
  int code0 = load_txt_file(f1, cloud1);
  int code1 = load_txt_file(f2, cloud2);
  int code2 = load_txt_file(f3, cloud3);

  std::cout << "Loading clouds " << cloud1->size() << " " << cloud2->size() << " " << cloud3->size() << std::endl;

  estimate_normal<pcl::PointXYZRGBNormal>(cloud1, input_params.normal_radius);
  estimate_normal<pcl::PointXYZRGBNormal>(cloud2, input_params.normal_radius);
  estimate_normal<pcl::PointXYZRGBNormal>(cloud3, input_params.normal_radius);


  pcl::FPFHEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::FPFHSignature33>::Ptr descr_est(new pcl::FPFHEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::FPFHSignature33>());
  
  descr_est->setRadiusSearch (input_params.descriptor_radius);

  // CLOUD 1
  descr_est->setInputCloud (cloud1);
  descr_est->setInputNormals (cloud1);
  
  std::cout << "Computing dense keypoints" << std::endl;
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr features (new pcl::PointCloud<pcl::FPFHSignature33> ());
  descr_est->compute (*features);
  
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  // CLOUD 2
  descr_est->setInputCloud (cloud2);
  descr_est->setInputNormals (cloud2);
  
  std::cout << "Computing dense keypoints" << std::endl;
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr features2 (new pcl::PointCloud<pcl::FPFHSignature33> ());
  descr_est->compute (*features2);

  pcl::KdTreeFLANN<pcl::FPFHSignature33> match_search;
  match_search.setInputCloud (features2);

  pcl::CorrespondencesPtr temp_correspondences(new pcl::Correspondences);
  pcl::CorrespondencesPtr temp_correspondences2(new pcl::Correspondences);

  pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal> rej;
  
  pcl::PCLPointCloud2::Ptr target_blob(new pcl::PCLPointCloud2), source_blob(new pcl::PCLPointCloud2);;
  toPCLPointCloud2(*cloud1, *source_blob);
  toPCLPointCloud2(*cloud2, *target_blob);

  pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> cor;
  cor.setInputCloud(features);
  cor.setInputTarget(features2);

  //cor.determineReciprocalCorrespondences(*temp_correspondences2);
  cor.determineCorrespondences(*temp_correspondences2);

  for (size_t i = 0; i < temp_correspondences2->size (); ++i) {
    int keypoint = temp_correspondences2->at(i).index_query;
    int match_keypoint = temp_correspondences2->at(i).index_match;

    pcl::PointXYZRGBNormal p = cloud1->points[keypoint], q = cloud2->points[match_keypoint];
    pcl::PointXYZ diff(p.x - q.x, p.y - q.y, p.z - q.z);
    float ddist = norm<pcl::PointXYZ>(diff);
    //std::cout << "KP " << keypoint << " " << match_keypoint << " " << "dist " << ddist <<  std::endl;
    // if(ddist < input_params.match_radius) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
    // {
    pcl::Correspondence corr (keypoint, match_keypoint, ddist);
    temp_correspondences->push_back(corr);
  }

  rej.setSourcePoints(source_blob);
  rej.setSourceNormals(source_blob);
  rej.setTargetPoints(target_blob);
  rej.setTargetNormals(target_blob);

  rej.setInputCorrespondences(temp_correspondences);

  rej.setMaximumIterations(input_params.ransac_max_iter);
  std::cout << "Inlier reject: " << input_params.match_radius << std::endl;
  rej.setInlierThreshold(input_params.match_radius);
  
  pcl::Correspondences correspondences;

  std::cout << "Initial correspondences " << temp_correspondences->size() << std::endl;
  rej.getCorrespondences(correspondences);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  std::cout << "Elapsed time: " << elapsed_secs << std::endl;

  std::cout << "Remaining correspondences " << correspondences.size() << std::endl;

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_X (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_Y (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_V (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  std::vector<int> indices(correspondences.size());
  
  std::ofstream outfile;
  outfile.open(result_file + "_indices.txt", std::fstream::out | std::fstream::trunc);

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result_cloud = cloud1;
  // if ( cloud1->size() < cloud2->size() )
  //     result_cloud = cloud2;

  pcl::copyPointCloud(*result_cloud, *new_X);
  pcl::copyPointCloud(*result_cloud, *new_Y);

  for (int i = 0; i < correspondences.size(); ++i)
  {

      int keypoint = correspondences[i].index_query;
      int match_keypoint = correspondences[i].index_match;

      //std::cout << "KP " << keypoint << " " << match_keypoint << " " << "dist " << correspondences[i].distance << std::endl;

      pcl::PointXYZRGBNormal p = cloud1->points[keypoint];
      pcl::PointXYZRGBNormal q = cloud2->points[match_keypoint];
      pcl::PointXYZRGBNormal v = cloud3->points[match_keypoint];
      
      new_X->points[i] = p;
      new_Y->points[i] = q;
      new_V->points.push_back(v);
      //indices.push_back(match_keypoint);

      outfile << keypoint+1 << " " << match_keypoint+1 << std::endl;
  }

  save_txt_file(result_file + "_new_V.txt", new_V);
  outfile.close();

  pcl::PointXYZRGBNormal nan;
  nan.x = NAN;
  nan.y = NAN;
  nan.z = NAN;
  for (int i = correspondences.size(); i < cloud1->size(); ++i) {
      new_X->points[i] = nan;
      new_Y->points[i] = nan;
  }

  save_txt_file(result_file + "_new_X.txt", new_X);
  save_txt_file(result_file + "_new_Y.txt", new_Y);
  
  return 0;
}
