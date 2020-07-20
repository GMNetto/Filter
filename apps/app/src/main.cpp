#include <iostream>
#include <unordered_map>
#include <fstream>
#include <chrono>

#include <pcl/common/common.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>

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

/** visualize keypoints **/
void keypoints_example(std::string f1, InputParams &input_params) {
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::io::loadPLYFile(f1, *cloud1);
    estimate_normal<pcl::PointXYZRGBNormal>(cloud1, input_params.normal_radius);
    
    Keypoint keypoints(input_params);
   
    std::shared_ptr<std::vector<int>> key_points = keypoints.get_keypoints(cloud1);

    std::cout << "number keypoints: " << key_points->size() << std::endl;

    for (int i=0; i < cloud1->size(); i++) {
      cloud1->points[i].r = 255;
      cloud1->points[i].g = 255;
      cloud1->points[i].b = 255;
    }

    for (int k: *key_points) {
      cloud1->points[k].r = 255;
      cloud1->points[k].g = 0;
      cloud1->points[k].b = 255;
    }
    
    //User interaction
    PointIteraction pI;
    PointIteraction::color = cloud1;
    PointIteraction::cloud = cloud1;

    CloudIteraction::cloud = cloud1;
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

  bool vec(false),
    ivec(false),
    cubes(false),
    crisp(false);
  std::string f1 = "",
      f2 = "",
      f3 = "";

  std::string tracklet_path = "";

  std::string dir = "/home/gustavo/DT3D/models/Block0.4/", result_file = "./results/";

  int start = 0, end = 4;

  if (pcl::console::find_argument (argc, argv, "-result-file") >= 0)
  {
    pcl::console::parse_argument (argc, argv, "-result-file", result_file);
  }
      
  if (pcl::console::find_argument (argc, argv, "-keypoints") >= 0)
  {
    std::cout << "Keypoints example" << std::endl;
    if (pcl::console::find_argument (argc, argv, "-f1") >= 0) {
      pcl::console::parse_argument (argc, argv, "-f1", f1);
    }
    keypoints_example(f1, input_params);
    return 0;
  }
  else if (pcl::console::find_argument (argc, argv, "-cubes") >= 0)
  {
    cubes = true ;
    std::cout << "Cubes example " << cubes << std::endl;
    readSequence(argc, argv, dir, tracklet_path, start, end);
  }
  else if (pcl::console::find_argument (argc, argv, "-vec") >= 0)
  {
    vec = true ;
    bool success = read2files(argc, argv, f1, f2);
    if (!success) {
      readSequence(argc, argv, dir, tracklet_path, start, end);
    } 
    std::cout << "Vec example " << vec << std::endl;
  }
  else if (pcl::console::find_argument (argc, argv, "-crispness") >= 0)
  {
    crisp = true ;
    read2files(argc, argv, f1, f2);
    std::cout << "Crispness" << std::endl;
    crispness_clouds(f1, f2, result_file, input_params);
    return 0;
  }
  else if (pcl::console::find_argument (argc, argv, "-ivec") >= 0)
  {
    ivec = true ;
    bool success = read2files(argc, argv, f1, f2);
    if (!success) {
      readSequence(argc, argv, dir, tracklet_path, start, end);
    } 
    std::cout << "IVec example " << vec << std::endl;
  }
  else
  {
    printUsage (argv[0]);
    return 0;
  }

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

  if (cubes)
  {
    KITTI kitti(input_params, seg_params, tracklet_path, dir, start, end, input_params.motion_bb);
    kitti.setInitialCloud();

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow (new pcl::PointCloud<pcl::PointXYZINormal>);

    // Cloud1 after icp projection and getting valid points
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    // Cloud2 after getting valid points
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    // Cloud1 projected with flow
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr icp_projected (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    // Motion from newC1 to newC2
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZINormal>);

    // Joined projection ICP and filter
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filter_projected (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    
    std::ofstream outfile, icpfile, outfile2, icpfile2;

    outfile.open(result_file + seg_params.segment_mode + "/cubes_" + input_params.kitti_tech + "_" + input_params.descriptor_tech + "_" + input_params.match_tech + "_" + input_params.focus_object + "_" + std::to_string(start) + "_" + std::to_string(end) + ".csv", std::fstream::out | std::fstream::trunc);
    icpfile.open(result_file + seg_params.segment_mode + "/cubes_" + input_params.kitti_tech + "_" + input_params.descriptor_tech + "_" + input_params.match_tech + "_" + input_params.focus_object + "_" + std::to_string(start) + "_" + std::to_string(end) + "_ICP.csv", std::fstream::out | std::fstream::trunc);
    outfile2.open(result_file + seg_params.segment_mode + "/cubes_" + input_params.kitti_tech + "_" + input_params.descriptor_tech + "_" + input_params.match_tech + "_" + input_params.focus_object + "_" + std::to_string(start) + "_" + std::to_string(end) + "2.csv", std::fstream::out | std::fstream::trunc);
    icpfile2.open(result_file + seg_params.segment_mode + "/cubes_" + input_params.kitti_tech + "_" + input_params.descriptor_tech + "_" + input_params.match_tech + "_" + input_params.focus_object + "_" + std::to_string(start) + "_" + std::to_string(end) + "_ICP2.csv", std::fstream::out | std::fstream::trunc);

    Spread sp(input_params);
    while (kitti.hasNext()) {
      icp_flow->clear();
      new_cloud1->clear();
      new_cloud2->clear();
      icp_projected->clear();
      new_cloud->clear();
      filter_projected->clear();      

      if (input_params.use_icp) {
        kitti.nextCloud(new_cloud1, icp_flow);
      } else {
        kitti.getObjects(new_cloud1);  
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        kitti.nextCloud(temp, icp_flow);
      }
      new_cloud2 = kitti.current_cloud;
      //kitti.getObjects(new_cloud2);

      project_points(icp_flow, icp_projected);

      std::cout << "Clouds to filter " << new_cloud1->size() << " " << new_cloud2->size() << std::endl;

      std::unordered_map<int, pcl::PointXYZ> vectors = sp.cloud_flow(new_cloud1,
                new_cloud2,
                new_cloud);

      join_flows(icp_flow, new_cloud);

      project_points(new_cloud, filter_projected);

      float icp_result = crispness(icp_projected, new_cloud2);
      float filter_result = crispness(filter_projected, new_cloud2);
      std::cout << "Icp result: " << icp_result << std::endl;
      std::cout << "Filter result: " << filter_result << std::endl;
      float icp_result2 = crispness3(icp_projected, new_cloud2);
      float filter_result2 = crispness3(filter_projected, new_cloud2);
      std::cout << "Icp result2: " << icp_result2 << std::endl;
      std::cout << "Filter result2: " << filter_result2 << std::endl;

      outfile << std::fixed << std::setprecision(4) << filter_result << std::endl;
      icpfile << std::fixed << std::setprecision(4) << icp_result << std::endl;  
      outfile2 << std::fixed << std::setprecision(4) << filter_result2 << std::endl;
      icpfile2 << std::fixed << std::setprecision(4) << icp_result2 << std::endl;  
    }
    
    outfile.close();
    icpfile.close();
    outfile2.close();
    icpfile2.close();
    return 0;
  }
  else if (vec == true)
  {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow (new pcl::PointCloud<pcl::PointXYZINormal>);
    bool bb = false;
    bool loaded = load_pair(f1, f2, cloud1, cloud2, input_params);
    
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud2 (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr s_cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr projected_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

    std::shared_ptr<KITTI> kitti;
    if (!loaded) {
      kitti = std::make_shared<KITTI>(input_params, seg_params, tracklet_path, dir, start, end, input_params.motion_bb);
      std::cout << "KITTI" << std::endl;
      kitti->setInitialCloud();

      std::cout << "Using ICP " << input_params.use_icp << std::endl;
      if (input_params.use_icp) {
        kitti->nextCloud(cloud1, icp_flow);
        std::cout << "CLOUD1 # " << cloud1->size() <<  std::endl;
      } else {
        kitti->getObjects(cloud1);  
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        kitti->nextCloud(temp, icp_flow);
      }
      cloud2 = kitti->current_cloud;
      std::cout << "CLOUD2 # " << cloud2->size() <<  std::endl;

      pcl::copyPointCloud(*cloud2, *s_cloud2);
      bb = true;
    } else if (input_params.kitti_tech == "c2ICP" || input_params.kitti_tech == "trICP" || input_params.kitti_tech == "pmICP") {
      KITTI kitti(input_params, seg_params);
      pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr projected_ (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
      std::cout << "ICP" << std::endl;
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      
      std::vector<int> points_cloud2;
      
      sample_cloud(cloud1, cloud2, points_cloud2, input_params.match_icp);
      
      //pcl::copyPointCloud(*cloud2, points_cloud2, *s_cloud2);
      pcl::copyPointCloud(*cloud2, *s_cloud2);
      std::cout << "C1: " << cloud1->size() << " C2: " << s_cloud2->size() << std::endl;
      
      
      kitti.icpBox(cloud1, s_cloud2, *projected_, icp_flow, transformed);
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
      std::cout << "Elapsed time: " << elapsed_secs << std::endl;
      pcl::copyPointCloud(*projected_, *cloud1);
    } else if (input_params.kitti_tech == "bbCPD") {
      KITTI kitti(input_params, seg_params);
      pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr projected_ (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
      std::cout << "CPD" << std::endl;
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      kitti.cpdBox(cloud1, cloud2, *projected_, icp_flow);
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
      std::cout << "Elapsed time: " << elapsed_secs << std::endl;
      pcl::copyPointCloud(*projected_, *cloud1);
    }

    Spread sp(input_params);
    std::unordered_map<int, pcl::PointXYZ> vectors;
    if (input_params.match_tech == "ICP") {
      KITTI kitti(input_params, seg_params);
      pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr projected_ (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
      std::cout << "ICP" << std::endl;
      std::clock_t begin = std::clock();
      kitti.icpBox(cloud1, cloud2, *projected_, new_cloud, transformed);
      std::clock_t end = std::clock();
      double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
      std::cout << "Elapsed time: " << elapsed_secs << std::endl;
      std::cout << "ICP2" << std::endl;
    } else if(input_params.kitti_tech == "SHOT_RANSAC") {
      KITTI kitti(input_params, seg_params);
      pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr projected_ (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
      std::cout << "C1: " << cloud1->size() << " C2: " << cloud2->size() << std::endl; 
      kitti.match_filter(cloud1, cloud2, *projected_, new_cloud);
    } else if (input_params.match_tech == "SKIP") {
      pcl::copyPointCloud(*cloud1, *new_cloud);
      for (int i = 0; i < new_cloud->size(); i++) {
        pcl::PointXYZINormal &vec = new_cloud->points[i];
        vec.normal_x = 0;
        vec.normal_y = 0;
        vec.normal_z = 0;
      }
    } else {
      vectors = sp.cloud_flow(cloud1, s_cloud2, new_cloud);
    }

    for (int i = 0; i < cloud1->size(); i++) {
        cloud1->points[i].r = 0;
        cloud1->points[i].g = 0;
        cloud1->points[i].b = 200;
    }

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr projected (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    project_points(new_cloud, projected);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr icp_projected (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    project_points(icp_flow, icp_projected);
    float icp_result = crispness(icp_projected, cloud2);
    float filter_result = crispness(projected, cloud2);
    std::cout << "Icp result: " << icp_result << std::endl;
    std::cout << "Filter result: " << filter_result << std::endl;

    float icp_result2 = crispness3(icp_projected, cloud2);
    float filter_result2 = crispness3(projected, cloud2);
    std::cout << "Icp result2: " << icp_result2 << std::endl;
    std::cout << "Filter result2: " << filter_result2 << std::endl;
    
    if (!input_params.use_GT) {
      flow_vis_loop(icp_flow);
      flow_vis_loop(new_cloud);
    }
    if (bb) {
      join_flows(icp_flow, new_cloud);
    }

    pcl::copyPointCloud(*cloud1, *projected_cloud);
    for (int i = 0; i < projected_cloud->size(); i++) {
      pcl::PointXYZRGBNormal &p = projected_cloud->points[i];
      pcl::PointXYZRGBNormal &p1 = cloud1->points[i];
      pcl::PointXYZRGBNormal &p2 = cloud2->points[i];
      pcl::PointXYZINormal &vec = new_cloud->points[i];
      p.x += vec.normal_x;
      p.y += vec.normal_y;
      p.z += vec.normal_z;
    }

    if (input_params.use_GT) {
      std::cout << "C1: " << cloud1->size() << " C2: " << cloud2->size() << "R: " << projected_cloud->size() << std::endl;
      std::cout << "Distance c1 -> c2: " << average_of_distance_GT(cloud2, cloud1) << std::endl;
      std::cout << "Distance proj -> c2: " << average_of_distance_GT(cloud2, projected_cloud) << std::endl;
    }

    if (result_file != "") {
      save_txt_file(result_file, projected_cloud);
    }

    if (!input_params.use_GT) {
      flow_vis_loop(new_cloud);
      for (int i = 0; i < cloud2->size(); i++) {
        cloud2->points[i].r = 40;
        cloud2->points[i].g = 40;
        cloud2->points[i].b = 40;
      }
      
      pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr merged (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
      project_points(new_cloud, cloud2, merged);
      viewer = rgbVis(merged);

      while (!viewer->wasStopped ())
      {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
      }
    }
    return 0;
  }
  else if (ivec == true)
  {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow (new pcl::PointCloud<pcl::PointXYZINormal>);
    bool bb = false;
    bool loaded = load_pair(f1, f2, cloud1, cloud2, input_params);
    if (!loaded) {
      KITTI kitti(input_params, seg_params, tracklet_path, dir, start, end, input_params.motion_bb);
   
      kitti.setInitialCloud();

      if (input_params.use_icp) {
        kitti.nextCloud(cloud1, icp_flow);
      } else {
        kitti.getObjects(cloud1);  
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        kitti.nextCloud(temp, icp_flow);
      }
      cloud2 = kitti.current_cloud;
      bb = true;
    }
    
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr projected_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    Spread sp(input_params);
    std::unordered_map<int, pcl::PointXYZ> vectors = sp.icloud_flow(cloud1,
              cloud2,
              new_cloud);
    // std::unordered_map<int, pcl::PointXYZ> vectors = sp.ioptimize_flow(cloud1,
    //           cloud2,
    //           new_cloud);

    if (bb) {
      join_flows(icp_flow, new_cloud);
    }

    flow_vis_loop(new_cloud);

    for (int i = 0; i < cloud2->size(); i++) {
        cloud2->points[i].r = 255;
        cloud2->points[i].g = 0;
        cloud2->points[i].b = 255;
    }

    project_points(new_cloud, cloud2);

    viewer = rgbVis(cloud2);

    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

    return 0;

  }
  return 0;
}
