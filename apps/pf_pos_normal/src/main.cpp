#include <iostream>
#include <unordered_map>
#include <fstream>
#include <chrono>

#include <pcl/io/obj_io.h>
#include <pcl/filters/uniform_sampling.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/log/core.hpp>

#include "visualize.hpp"
#include "metrics.hpp"
#include "util.hpp"
#include "pf.hpp"
#include "pfpoint.hpp"
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

/** filter point position with PF **/
void pf_position(std::string f1, std::string f2, InputParams &input_params, std::string result_file) {
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr original_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr custom_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    bool loaded = load_pair(f1, f2, original_cloud, custom_cloud, input_params);
    std::cout << "Loaded clouds" << std::endl;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    transfer_normals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(original_cloud, custom_cloud);
    std::cout << "Transfered normals " << custom_cloud->size() << " " << original_cloud->size() << std::endl;
    pcl::copyPointCloud(*custom_cloud, *result);
    std::cout << "Initial avg point distance: " << average_point_distance<pcl::PointXYZRGBNormal>(custom_cloud, original_cloud) << std::endl;
    std::vector<int> order_visit(custom_cloud->size());
    pf_3D_point(custom_cloud, result, input_params.number_init, true, input_params.sigma_s, input_params.sigma_r, input_params.sigma_d);
    std::cout << "Result avg point distance: " << average_point_distance<pcl::PointXYZRGBNormal>(result, original_cloud) << std::endl;
    
    pcl::PolygonMesh mesh;
    pcl::io::loadOBJFile(f1, mesh);
    for (int i = 0; i < result->size(); i++) {
      pcl::PointXYZRGBNormal &p = result->points[i];
      p.r = 255;
      p.g = 0;
      p.b = 255;
    }
    pcl::toPCLPointCloud2(*result, mesh.cloud);

    std::cout << "Save mesh at: " << result_file << " " << pcl::io::saveOBJFile(result_file, mesh) << std::endl;
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = meshVis(mesh);

    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
}

/**display and save sample cloud**/
void sample_example(std::string f1, InputParams &input_params, std::string result_file) {
   pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    load_cloud(f1, cloud1, input_params);
    estimate_normal<pcl::PointXYZRGBNormal>(cloud1, input_params.normal_radius);

    pcl::UniformSampling<pcl::PointXYZRGBNormal> uniform_detector;
    uniform_detector.setInputCloud(cloud1);
    uniform_detector.setRadiusSearch(input_params.uniform_radius);

    std::cout << "Starting cloud " << cloud1->size() << std::endl;

    std::cout << "Uniform radius " << input_params.uniform_radius << std::endl;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    uniform_detector.filter (*temp);

    std::cout << "Result cloud " << temp->size() << std::endl;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudColored (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::copyPointCloud(*temp, *cloudColored);


    for (int i=0; i < temp->size(); i++) {
      cloudColored->points[i].r = 255;
      cloudColored->points[i].g = 0;
      cloudColored->points[i].b = 0;
    }

    
    std::string prefix = result_file + "_" + std::to_string(input_params.uniform_radius), txt = prefix + ".txt", pcd = prefix + ".pcd";
    save_txt_file(txt, cloudColored);
    pcl::io::savePCDFileASCII(pcd, *temp);
    
    //User interaction
    PointIteraction pI;
    PointIteraction::color = cloudColored;
    PointIteraction::cloud = cloudColored;

    CloudIteraction::cloud = cloudColored;
    std::shared_ptr<CloudIteraction> cI = std::make_shared<CloudIteraction>();   

    PointIteraction::cI = cI;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = pI.interactionCustomizationVis();

    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
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

  bool pf(false);
  std::string f1 = "",
      f2 = "",
      f3 = "";

  bool rot_trans = false;

  std::string tracklet_path = "";

  std::string dir = "/home/gustavo/DT3D/models/Block0.4/", result_file = "./results/";

  int start = 0, end = 4;

  if (pcl::console::find_argument (argc, argv, "-result-file") >= 0)
  {
    pcl::console::parse_argument (argc, argv, "-result-file", result_file);
  }
      
  if (pcl::console::find_argument (argc, argv, "-pf") >= 0)
  {
    pf = true ;
    read2files(argc, argv, f1, f2);
    std::cout << "PF example " << pf << std::endl;
  }
  else if (pcl::console::find_argument (argc, argv, "-pfpoint") >= 0)
  {
    read2files(argc, argv, f1, f2);
    std::cout << "PF point example" << std::endl;
    pf_position(f1, f2, input_params, result_file);
    return 0;
  }
  else if (pcl::console::find_argument (argc, argv, "-sample") >= 0)
  {
    std::cout << "Sampling example" << std::endl;
    if (pcl::console::find_argument (argc, argv, "-f1") >= 0) {
      pcl::console::parse_argument (argc, argv, "-f1", f1);
    }
    sample_example(f1, input_params, result_file);
    return 0;
  }
  else
  {
    printUsage (argv[0]);
    return 0;
  }

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

  if (pf)
  {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr original_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr custom_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    bool loaded = load_pair(f1, f2, original_cloud, custom_cloud, input_params);
    //pcl::io::loadPLYFile(f1, *original_cloud);
    //pcl::io::loadPLYFile(f2, *custom_cloud);
    pcl::copyPointCloud(*custom_cloud, *result);
    std::cout << "Loaded clouds " << custom_cloud->size() << " " << original_cloud->size() << std::endl;
    std::cout << "Initial avg point distance: " << average_angle<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(custom_cloud, original_cloud) << std::endl;
    std::vector<int> order_visit(custom_cloud->size());
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result_points (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    //pcl::copyPointCloud(*custom_cloud, *result_points);
    pcl::copyPointCloud(*result, *result_points);
    for (int i = 0; i < input_params.number_init; i++) {
      // Filter normals
      pf_3D_normal(result, result_points, 2, true, input_params.sigma_s, input_params.sigma_r, input_params.sigma_d);
      normalize_normals<pcl::PointXYZRGBNormal>(result);

      //estimate_normal<pcl::PointXYZRGBNormal>(result, input_params.normal_radius);

      // Filter Points
      pcl::copyPointCloud(*result_points, *result);
      pf_3D_point(result, result_points, 1, true, input_params.sigma_s, input_params.sigma_r, input_params.sigma_d);
      pcl::copyPointCloud(*result_points, *result);  
    }
    /*
    // Filter normals
    pf_3D_normal(custom_cloud, result, input_params.number_init, true, input_params.sigma_s, input_params.sigma_r, input_params.sigma_d);
    normalize_normals<pcl::PointXYZRGBNormal>(result);

    // Filter Points
    pcl::copyPointCloud(*result, *result_points);
    pf_3D_point(result, result_points, input_params.number_init, true, input_params.sigma_s, input_params.sigma_r, input_params.sigma_d);

    // Filter Normals
    pcl::copyPointCloud(*result_points, *result);
    pf_3D_normal(result_points, result, input_params.number_init, true, input_params.sigma_s, input_params.sigma_r, input_params.sigma_d);
    normalize_normals<pcl::PointXYZRGBNormal>(result);

    // Filter Points
    pcl::copyPointCloud(*result, *result_points);
    pf_3D_point(result, result_points, input_params.number_init, true, input_params.sigma_s, input_params.sigma_r, input_params.sigma_d);
    */
    std::cout << "Result avg point distance: " << average_angle<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(result, original_cloud) << std::endl;
    color_normals2<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(result, result);

    pcl::PolygonMesh mesh;
    pcl::io::loadOBJFile(f1,mesh);

    
    pcl::toPCLPointCloud2(*result, mesh.cloud);
    std::cout << "Save mesh at: " << result_file << " " << pcl::io::saveOBJFile(result_file, mesh) << std::endl;
    viewer = meshVis(mesh);
  }

  //--------------------
  // -----Main loop-----
  //--------------------
  //The two windows is a known problem
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
}
