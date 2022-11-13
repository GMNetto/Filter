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
#include "permutohedral.h"

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

  bool vec(false), perm(false), upvec(false), upperm(false);
  std::string f1 = "", f2 = "", f3 = "", f4 = "", result_file = "";
  
  if (pcl::console::find_argument (argc, argv, "-result-file") >= 0)
  {
    pcl::console::parse_argument (argc, argv, "-result-file", result_file);
  }
      
  if (pcl::console::find_argument (argc, argv, "-vec") >= 0)
  {
    vec = true ;
    read2files(argc, argv, f1, f2);
    std::cout << "Vec example " << vec << std::endl;
  } else if (pcl::console::find_argument (argc, argv, "-perm") >= 0) {
    perm = true ;
    read2files(argc, argv, f1, f2);
    std::cout << "Perm example " << vec << std::endl;
  } else if (pcl::console::find_argument (argc, argv, "-upvec") >= 0) {
    upvec = true ;
    if (pcl::console::find_argument (argc, argv, "-f1") >= 0) {
      pcl::console::parse_argument (argc, argv, "-f1", f1);
    }
    if (pcl::console::find_argument (argc, argv, "-f2") >= 0) {
      pcl::console::parse_argument (argc, argv, "-f2", f2);
    }
    if (pcl::console::find_argument (argc, argv, "-f3") >= 0) {
      pcl::console::parse_argument (argc, argv, "-f3", f3);
    }
    if (pcl::console::find_argument (argc, argv, "-f4") >= 0) {
      pcl::console::parse_argument (argc, argv, "-f4", f4);
    }
    std::cout << "Upvec example " << vec << std::endl;
  } else if (pcl::console::find_argument (argc, argv, "-upperm") >= 0) {
    upperm = true ;
    if (pcl::console::find_argument (argc, argv, "-f1") >= 0) {
      pcl::console::parse_argument (argc, argv, "-f1", f1);
    }
    if (pcl::console::find_argument (argc, argv, "-f2") >= 0) {
      pcl::console::parse_argument (argc, argv, "-f2", f2);
    }
    if (pcl::console::find_argument (argc, argv, "-f3") >= 0) {
      pcl::console::parse_argument (argc, argv, "-f3", f3);
    }
    if (pcl::console::find_argument (argc, argv, "-f4") >= 0) {
      pcl::console::parse_argument (argc, argv, "-f4", f4);
    }
    std::cout << "Upperm example " << upperm << std::endl;
  }
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

  if (vec) {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr object_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr flow_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    load_pair(f1, f2, object_cloud, flow_cloud, input_params);
    
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::copyPointCloud(*object_cloud, *vecs);

    for (int i = 0; i < flow_cloud->points.size(); i++) {
        pcl::PointXYZINormal &v = vecs->points[i];
        pcl::PointXYZRGBNormal &p = flow_cloud->points[i];

        v.normal_x = p.x;
        v.normal_y = p.y;
        v.normal_z = p.z;
    }

    SpatialFilter sp(input_params);

    // Get rundom points to start
    std::vector<int> valid_keypoints;
    valid_keypoints.push_back(0);
    valid_keypoints.push_back(100);
    valid_keypoints.push_back(1000);

    sp.pf_3D_vec(object_cloud, vecs, new_cloud, valid_keypoints);

    if (result_file != "") {
      // Save into file with vec_x vec_y vec_z
        std::ofstream outfile;

        std::cout << "Saving " << result_file << std::endl;
        outfile.open(result_file, std::fstream::out | std::fstream::trunc);
        for (int i = 0; i < new_cloud->points.size(); i++) {
            pcl::PointXYZINormal &p = new_cloud->points[i];
            outfile << p.normal_x << " " << p.normal_y << " " << p.normal_z << std::endl;
        }
        outfile.close();   
    }
  } else if (upvec) {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sampled_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sampled_flow (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    load_pair(f1, f2, sampled_cloud, sampled_flow, input_params);
    load_pair(f3, f4, cloud1, cloud2, input_params);

    std::shared_ptr<std::vector<int>> key_points = get_flownet_keypoints(cloud1, sampled_cloud);
    
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::copyPointCloud(*cloud1, *vecs);

    for (int i=0; i < vecs->size(); i++) {
        vecs->points[i].normal_x = 0;
        vecs->points[i].normal_y = 0;
        vecs->points[i].normal_z = 0;
        vecs->points[i].intensity = 0;
    }

    for (int i = 0; i < key_points->size(); i++) {
      int idx = key_points->at(i);
      pcl::PointXYZRGBNormal &vec = sampled_flow->points[i];
      pcl::PointXYZINormal &v = vecs->points[idx];
      v.normal_x = vec.x;
      v.normal_y = vec.y;
      v.normal_z = vec.z;
      v.intensity = 1;
    }

    SpatialFilter sp(input_params);

    // Get rundom points to start
    std::vector<int> valid_keypoints;
    valid_keypoints.push_back(0);
    valid_keypoints.push_back(100);
    valid_keypoints.push_back(1000);

    sp.pf_3D_vec(cloud1, vecs, new_cloud, valid_keypoints);
    normalize_colors<pcl::PointXYZINormal>(new_cloud, 0.0001);
    if (result_file != "") {
      // Save into file with vec_x vec_y vec_z
        std::ofstream outfile;

        std::cout << "Saving " << result_file << std::endl;
        outfile.open(result_file, std::fstream::out | std::fstream::trunc);
        for (int i = 0; i < new_cloud->points.size(); i++) {
            pcl::PointXYZINormal &p = new_cloud->points[i];
            outfile << p.normal_x << " " << p.normal_y << " " << p.normal_z << std::endl;
        }
        outfile.close();   
    }
  } else if (perm) {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr object_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr flow_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    load_pair(f1, f2, object_cloud, flow_cloud, input_params);
    
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::copyPointCloud(*object_cloud, *vecs);

    for (int i = 0; i < flow_cloud->points.size(); i++) {
        pcl::PointXYZINormal &v = vecs->points[i];
        pcl::PointXYZRGBNormal &p = flow_cloud->points[i];

        v.normal_x = p.x;
        v.normal_y = p.y;
        v.normal_z = p.z;
    }

    std::vector<float> feature, feature2, out, normalize, out_n;

    feature.resize ((vecs->points).size() * 6);
    feature2.resize ((vecs->points).size() * 3);
    normalize.resize ((vecs->points).size() * 1);
    out.resize ((vecs->points).size() * 3);
    out_n.resize ((vecs->points).size() * 1);

    float sigma_s = 1.0;
    float sigma_r = 0.3;
    for (int i=0; i < (object_cloud->points).size(); i++)
    {
        feature[i*6]=(object_cloud->points[i].x)/sigma_s;
        feature[i*6+1]=(object_cloud->points[i].y)/sigma_s;
        feature[i*6+2]=(object_cloud->points[i].z)/sigma_s;

        feature[i*6+3]=(object_cloud->points[i].normal_x)/sigma_r;
        feature[i*6+4]=(object_cloud->points[i].normal_y)/sigma_r;
        feature[i*6+5]=(object_cloud->points[i].normal_z)/sigma_r;

        // feature2[i*3]=(vecs->points[i]).normal_x;
        // feature2[i*3+1]=vecs->points[i].normal_y;
        // feature2[i*3+2]=vecs->points[i].normal_z;
        feature2[i*3]=(object_cloud->points[i].x);
        feature2[i*3+1]=(object_cloud->points[i].y);
        feature2[i*3+2]=(object_cloud->points[i].z);
        normalize[i]=1;
    } 

    pcl::Permutohedral p;  
    std::cout << "Init" << std::endl;
    p.init(feature,6,(vecs->points).size());
    //p.debug();
    std::cout << "Comp" << std::endl;
    p.compute(out,feature2,3,0,0,(vecs->points).size(),(vecs->points).size());
    p.compute(out_n,normalize,1,0,0,(vecs->points).size(),(vecs->points).size());
    std::cout << "End" << std::endl;

    if (result_file != "") {
      // Save into file with vec_x vec_y vec_z
        std::ofstream outfile;

        std::cout << "Saving " << result_file << std::endl;
        outfile.open(result_file, std::fstream::out | std::fstream::trunc);
        for (int i = 0; i < vecs->points.size(); i++) {
            float den = out_n[i];
            outfile << out[i*3]/den << " " << out[i*3+1]/den << " " << out[i*3+2]/den << std::endl;
        }
        outfile.close();   
    }
  } else if (upperm) {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sampled_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sampled_flow (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    load_pair(f1, f2, sampled_cloud, sampled_flow, input_params);
    load_pair(f3, f4, cloud1, cloud2, input_params);

    std::chrono::steady_clock::time_point begin_knn = std::chrono::steady_clock::now();
    std::shared_ptr<std::vector<int>> key_points = get_flownet_keypoints(cloud1, sampled_cloud);
    std::chrono::steady_clock::time_point end_knn = std::chrono::steady_clock::now();
    double elapsed_secs_knn = std::chrono::duration_cast<std::chrono::milliseconds>(end_knn - begin_knn).count();
    std::cout << "Elapsed filter time: " << elapsed_secs_knn << std::endl;   


    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::copyPointCloud(*cloud1, *vecs);

    for (int i=0; i < vecs->size(); i++) {
        vecs->points[i].normal_x = 0;
        vecs->points[i].normal_y = 0;
        vecs->points[i].normal_z = 0;
        vecs->points[i].intensity = 0;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < key_points->size(); i++) {
      int idx = key_points->at(i);
      pcl::PointXYZRGBNormal &vec = sampled_flow->points[i];
      pcl::PointXYZINormal &v = vecs->points[idx];
      v.normal_x = vec.x;
      v.normal_y = vec.y;
      v.normal_z = vec.z;
      v.intensity = 1;
    }

    std::vector<float> feature, feature2, out, normalize, out_n;

    feature.resize ((vecs->points).size() * 3);
    feature2.resize ((vecs->points).size() * 4);
    normalize.resize ((vecs->points).size() * 1);
    out.resize ((vecs->points).size() * 4);
    out_n.resize ((vecs->points).size() * 1);

    float sigma_s = input_params.sigma_s;
    float sigma_r = input_params.sigma_r;
    for (int i=0; i < (cloud1->points).size(); i++)
    {
        feature[i*3]=(cloud1->points[i].x)/sigma_s;
        feature[i*3+1]=(cloud1->points[i].y)/sigma_s;
        feature[i*3+2]=(cloud1->points[i].z)/sigma_s;

        // feature[i*6+3]=(cloud1->points[i].normal_x)/sigma_r;
        // feature[i*6+4]=(cloud1->points[i].normal_y)/sigma_r;
        // feature[i*6+5]=(cloud1->points[i].normal_z)/sigma_r;

        feature2[i*4]=(vecs->points[i]).normal_x;
        feature2[i*4+1]=vecs->points[i].normal_y;
        feature2[i*4+2]=vecs->points[i].normal_z;
        feature2[i*4+3]=vecs->points[i].intensity;
        // feature2[i*3]=(object_cloud->points[i].x);
        // feature2[i*3+1]=(object_cloud->points[i].y);
        // feature2[i*3+2]=(object_cloud->points[i].z);
        normalize[i]=1;
    } 

    pcl::Permutohedral p;  

    Eigen::Matrix<float, 3, Eigen::Dynamic> src(3, cloud1->size());
    for (size_t i = 0; i < cloud1->size(); ++i) {
        src.col(i) << cloud1->points[i].x/sigma_s, cloud1->points[i].y/sigma_s, cloud1->points[i].z/sigma_s;
    }

    std::chrono::steady_clock::time_point begin_init = std::chrono::steady_clock::now();
    Permutohedral perm;
    perm.init(src);
    std::chrono::steady_clock::time_point end_init = std::chrono::steady_clock::now();
    double elapsed_secs_init = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - begin_init).count();
    std::cout << "Elapsed init time: " << elapsed_secs_init << std::endl;   


    std::cout << "Init" << std::endl;

    begin_init = std::chrono::steady_clock::now();
    p.init(feature,3,(vecs->points).size());
    end_init = std::chrono::steady_clock::now();
    elapsed_secs_init = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - begin_init).count();
    std::cout << "Elapsed init time 2: " << elapsed_secs_init << std::endl;   

    //p.debug();
    std::cout << "Comp" << std::endl;
    p.compute(out,feature2,4,0,0,(vecs->points).size(),(vecs->points).size());
    p.compute(out_n,normalize,1,0,0,(vecs->points).size(),(vecs->points).size());
    std::cout << "End" << std::endl;

    pcl::copyPointCloud(*cloud1, *new_cloud);

    for (int i = 0; i < new_cloud->points.size(); i++) {
      float den = out_n[i];

      pcl::PointXYZINormal &r = new_cloud->points[i];
      r.normal_x = out[i*4]/den;
      r.normal_y = out[i*4+1]/den;
      r.normal_z = out[i*4+2]/den;
      r.intensity = out[i*4+3]/den;
    }

    normalize_colors<pcl::PointXYZINormal>(new_cloud, 0.0001);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed filter time: " << elapsed_secs << std::endl;   

    if (result_file != "") {
      // Save into file with vec_x vec_y vec_z
        std::ofstream outfile;

        std::cout << "Saving " << result_file << " " << new_cloud->points.size() <<std::endl;
        outfile.open(result_file, std::fstream::out | std::fstream::trunc);
        for (int i = 0; i < new_cloud->points.size(); i++) {
            pcl::PointXYZINormal &r = new_cloud->points[i];
            outfile << r.normal_x << " " << r.normal_y << " " << r.normal_z << std::endl;
        }
        std::cout << "Saving2 " << result_file << std::endl;
        outfile.close();   
    }
  }
  // Take three files as input output_x, output_y (calculate vectors)
  // Take output file, filter and 
  return 0;
}


