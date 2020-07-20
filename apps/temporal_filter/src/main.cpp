#include <iostream>
#include <unordered_map>
#include <fstream>
#include <chrono>

#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "visualize.hpp"
#include "metrics.hpp"
#include "util.hpp"
#include "pf_vector.hpp"
#include "normal.hpp"
#include "filterPF.hpp"
#include "temporalfilter.hpp"
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

int main(int argc, char **argv) {
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

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

  bool temporal(false), seq(false);
  std::string f1 = "", f2 = "", f3 = "";

  std::string tracklet_path = "";

  std::string dir = "/home/gustavo/DT3D/models/Block0.4/", result_file = "./results/";

  int start = 0, end = 4;

  if (pcl::console::find_argument (argc, argv, "-result-file") >= 0)
  {
    pcl::console::parse_argument (argc, argv, "-result-file", result_file);
  }
  
  if (pcl::console::find_argument (argc, argv, "-temporal") >= 0)
  {
    temporal = true ;
    std::cout << "Temporal example " << temporal << std::endl;
    readSequence(argc, argv, dir, tracklet_path, start, end);
  }
  else if (pcl::console::find_argument (argc, argv, "-seq") >= 0)
  {
    seq = true ;
    readSequence(argc, argv, dir, tracklet_path, start, end);
    std::cout << "Sequence example " << dir << " " << start << " " << end << std::endl;
  }
  else
  {
    printUsage (argv[0]);
    return 0;
  }

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

  if (temporal)
  {
    std::string temporal_config = "temp_config.json";
    if (pcl::console::find_argument (argc, argv, "-temp_config") >= 0) {
      pcl::console::parse_argument (argc, argv, "-temp_config", temporal_config);
    }

    boost::property_tree::ptree pt_temp;
    boost::property_tree::read_json(temporal_config, pt_temp);
    
    //Loading cloud files
    std::vector<std::string> cloud_files;
    std::cout << "Reading dir: " << dir << std::endl;
    read_directory(dir, cloud_files);
    std::sort(cloud_files.begin(), cloud_files.end(), compareNat);
    for (int i = start; i < end; i++) {
        std::cout << "file: " << cloud_files[i] << std::endl;
    }

    std::cout << "Loading" << std::endl;
    // new_cloud1 has the first cloud position
    std::string fullpath = dir + cloud_files[start];
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    load_cloud(fullpath, new_cloud1, input_params);

    std::cout << "Loading" << std::endl;
    // new_cloud2 has the second cloud position
    fullpath = dir + cloud_files[start + 1];
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    load_cloud(fullpath, new_cloud2, input_params);

    // new_cloud has the filter result
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr null_flow (new pcl::PointCloud<pcl::PointXYZINormal>);

    Spread sp(input_params);

    //User interaction
    FlowIteraction fI;
    FlowIteraction::vectors = new_cloud;
    FlowIteraction::cloud = new_cloud1;

    std::cout << "Filtering" << std::endl;
    std::unordered_map<int, pcl::PointXYZ> vectors = sp.cloud_flow(new_cloud1,
              new_cloud2,
              new_cloud);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = fI.flowVis(new_cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_result (new pcl::PointCloud<pcl::PointXYZRGB>);

    viewer->registerPointPickingCallback (FlowIteraction::vectors_pick, (void*)viewer.get());
    viewer->registerKeyboardCallback (FlowIteraction::update_cloud, (void*)viewer.get());
    
    float temp_sigma_s = pt_temp.get<float>("sigma_s"), temp_sigma_r = pt_temp.get<float>("sigma_r"),
     temp_radius = pt_temp.get<float>("radius"), spatial_sigma_s = pt_temp.get<float>("sigma_s_spatial"),
     spatial_sigma_r = pt_temp.get<float>("sigma_r_spatial");

    null_flow->resize(new_cloud->size());
    TemporalFilter temporal_filter(new_cloud1, new_cloud, null_flow, 0, temp_radius, spatial_sigma_r, spatial_sigma_s, temp_sigma_r, temp_sigma_s);

    int counter = 2;
    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
      boost::mutex::scoped_lock updateLock(FlowIteraction::updateModelMutex);
      if(FlowIteraction::update)
      {
          pcl::copyPointCloud(*new_cloud2, *new_cloud1); 
          //new_cloud1->clear();
          new_cloud2->clear();
          
          fullpath = dir + cloud_files[start + counter];
          std::cout << "file: " << fullpath << std::endl;
          load_cloud(fullpath, new_cloud2, input_params);

          // Project with previous flow
          pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr merged_p (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
          pcl::copyPointCloud(*new_cloud1, *merged_p);
          //project_points(new_cloud, new_cloud1, merged_p);  
          //temporal_filter.warp_base(new_cloud1, new_cloud, merged_p);

          new_cloud->clear();

          vectors = sp.cloud_flow(merged_p,
                new_cloud2,
                new_cloud);
                
          null_flow->resize(new_cloud->size());
          temporal_filter.filter_temporal(new_cloud1,
                new_cloud2,
                new_cloud,
                null_flow);

          std::cout << "Getting last flow" << std::endl;
          pcl::PointCloud<pcl::PointXYZINormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZINormal>);
          pcl::PointCloud<pcl::PointXYZINormal>::Ptr res = temporal_filter.get_last_flow();
          pcl::copyPointCloud(*res, *temp);
          std::cout << "Got last flow" << std::endl;

          pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr projected (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
          project_points(new_cloud, projected);
          pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr projected2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
          project_points(temp, projected2);

          float icp_result = crispness3(new_cloud1, new_cloud2);
          float filter_result = crispness3(projected, new_cloud2);
          float filter_result2 = crispness3(projected2, new_cloud2);
          std::cout << "Icp result: " << icp_result << std::endl;
          std::cout << "Filter result: " << filter_result << std::endl;
          std::cout << "Filter result: " << filter_result2 << std::endl;

          pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr merged (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
          project_points(new_cloud, new_cloud2, merged);
          
          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(merged);
          viewer->updatePointCloud<pcl::PointXYZRGBNormal> (merged, rgb, "sample cloud");
          viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
          FlowIteraction::set_camera(viewer);
          
          
          std::cout << "updated cloud2" << std::endl;
          FlowIteraction::update = false;
          counter+=1;
      }
      updateLock.unlock();
    }

    return 0;
  }
  else if (seq == true)
  {
    std::string temporal_config = "temp_config.json";
    if (pcl::console::find_argument (argc, argv, "-temp_config") >= 0) {
      pcl::console::parse_argument (argc, argv, "-temp_config", temporal_config);
    }

    boost::property_tree::ptree pt_temp;
    boost::property_tree::read_json(temporal_config, pt_temp);

    KITTI kitti(input_params, seg_params, tracklet_path, dir, start, end, input_params.motion_bb);
   
    kitti.setInitialCloud();

    // icp_flow has the icp flow
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow (new pcl::PointCloud<pcl::PointXYZINormal>);

    // new_cloud1 has the first cloud position
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    kitti.getObjects(new_cloud1);  

    // new_cloud1 warped with icp_flow
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr icp_warped (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    kitti.nextCloud(icp_warped, icp_flow);

    // new_cloud2 has the second cloud position
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    kitti.getObjects(new_cloud2);

    // new_cloud has the filter result
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZINormal>);

    Spread sp(input_params);

    //User interaction
    FlowIteraction fI;
    FlowIteraction::vectors = new_cloud;
    FlowIteraction::cloud = new_cloud1;

    std::unordered_map<int, pcl::PointXYZ> vectors = sp.cloud_flow(icp_warped,
              new_cloud2,
              new_cloud);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = fI.flowVis(new_cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_result (new pcl::PointCloud<pcl::PointXYZRGB>);

    viewer->registerPointPickingCallback (FlowIteraction::vectors_pick, (void*)viewer.get());

    //viewer->registerKeyboardCallback (camera_pos, (void*)viewer.get ());

    viewer->registerKeyboardCallback (FlowIteraction::update_cloud, (void*)viewer.get());

    // joins both flows into new_cloud
    //join_flows(icp_flow, new_cloud);

    float temp_sigma_s = pt_temp.get<float>("sigma_s"), temp_sigma_r = pt_temp.get<float>("sigma_r"),
     temp_radius = pt_temp.get<float>("radius"), spatial_sigma_s = pt_temp.get<float>("sigma_s_spatial"),
     spatial_sigma_r = pt_temp.get<float>("sigma_r_spatial");
    TemporalFilter temporal_filter(new_cloud1, new_cloud, icp_flow, 0, temp_radius, spatial_sigma_r, spatial_sigma_s, temp_sigma_r, temp_sigma_s);

    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
      boost::mutex::scoped_lock updateLock(FlowIteraction::updateModelMutex);
      if(FlowIteraction::update)
      {
          icp_flow->clear();
          icp_warped->clear();
          new_cloud1->clear();
          new_cloud2->clear();
          new_cloud->clear(); 

          kitti.getObjects(new_cloud1);  
          kitti.nextCloud(icp_warped, icp_flow);

          kitti.getObjects(new_cloud2);

          vectors = sp.cloud_flow(icp_warped,
                new_cloud2,
                new_cloud);

          // joins both flows into new_cloud
          //join_flows(icp_flow, new_cloud);
          temporal_filter.filter_temporal(new_cloud1,
                new_cloud2,
                new_cloud,
                icp_flow);

          std::cout << "Getting last flow" << std::endl;
          pcl::PointCloud<pcl::PointXYZINormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZINormal>);
          pcl::PointCloud<pcl::PointXYZINormal>::Ptr res = temporal_filter.get_last_flow();
          pcl::copyPointCloud(*res, *temp);
          std::cout << "Got last flow" << std::endl;

          pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr projected (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
          join_flows(icp_flow, new_cloud);
          project_points(new_cloud, projected);
          pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr icp_projected (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
          project_points(icp_flow, icp_projected);
          pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr projected2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
          join_flows(icp_flow, temp);
          project_points(temp, projected2);

          float icp_result = crispness3(icp_projected, new_cloud2);
          float filter_result = crispness3(projected, new_cloud2);
          float filter_result2 = crispness3(projected2, new_cloud2);
          std::cout << "Icp result: " << icp_result << std::endl;
          std::cout << "Filter result: " << filter_result << std::endl;
          std::cout << "Filter result: " << filter_result2 << std::endl;

          pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr merged (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
          project_points(temp, new_cloud2, merged);
          
          //FlowIteraction::get_colored_cloud(temp, colored_result);
	        //color_normals2<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(cloud3, colored_result);

          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(merged);
          viewer->updatePointCloud<pcl::PointXYZRGBNormal> (merged, rgb, "sample cloud");
          viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
          FlowIteraction::set_camera(viewer);
          
          
          std::cout << "updated cloud2" << std::endl;
          FlowIteraction::update = false;
      }
      updateLock.unlock();
    }

    return 0;
  }
  return 0;
}
