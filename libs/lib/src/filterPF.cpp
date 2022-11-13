#include "filterPF.hpp"
#include "util.hpp"
#include "pf_vector.hpp"
#include "keypoints.hpp"
#include "visualize.hpp"
#include "match.hpp"

#include <string>
#include <unordered_map>
#include <ctime>
#include <chrono>

#include <pcl/common/common.h>
#include <pcl/common/common_headers.h>
#include <pcl/filters/bilateral.h>
#include "normal.hpp"

void
FlowFilter::flow_descriptor(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                std::unordered_map<int, pcl::PointXYZ> &vectors,
                std::unordered_map<int, int> &prev2next) {
    
    this->matcherSHOT->local_correspondences(key_points, cloud1, cloud2, vectors, prev2next);
}

void
FlowFilter::flow_neighborhood(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                std::unordered_map<int, pcl::PointXYZ> &vectors,
                std::unordered_map<int, int> &prev2next) {
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input1_keypoints (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());

    //pcl::copyPointCloud (*cloud1, *key_points, *input1_keypoints); 
    
    //save_txt_file("keypoints.txt", input1_keypoints);

    this->matcher->local_correspondences(key_points, cloud1, cloud2, vectors, prev2next);
}

void
FlowFilter::flow_cpd(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                std::unordered_map<int, pcl::PointXYZ> &vectors) {
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input1_keypoints (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());

    pcl::copyPointCloud (*cloud1, *key_points, *input1_keypoints); 
    
    std::unordered_map<int, int> matches;
    this->matcherCPD->local_correspondences(key_points, cloud1, cloud2, vectors, matches);
}

void
FlowFilter::color(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    

    // Filter to spread correspondences
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::copyPointCloud(*cloud1, *vecs);

    for (int i=0; i < vecs->size(); i++) {
        vecs->points[i].normal_x = 0;
        vecs->points[i].normal_y = 0;
        vecs->points[i].normal_z = 0;
        vecs->points[i].intensity = 0;
    }
    for (int it: *key_points) {
        //std::cout << "Assign " << it.first << " " << c_vector << std::endl;
        vecs->points[it].normal_x = 1;
        vecs->points[it].normal_y = 0;
        vecs->points[it].normal_z = 0;
        vecs->points[it].intensity = 1;
    }

    std::vector<int> valid_keypoints;
    valid_keypoints.reserve(key_points->size());
    for (int i = 0 ; i < key_points->size(); i++) {
        valid_keypoints.push_back(key_points->at(i));
    }

    if (this->match_tech == "PERM") {
        perm_filter->filter(cloud1, vecs, new_cloud);
        normalize_colors<pcl::PointXYZINormal>(new_cloud, 0.0001);
        return;
    }
    spatial_filter->pf_3D_vec(cloud1, vecs, new_cloud, valid_keypoints);
    normalize_colors<pcl::PointXYZINormal>(new_cloud, 0.0001);
    
}

std::unordered_map<int, pcl::PointXYZ>
FlowFilter::flow4(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    
    std::vector<int> prev2next(key_points->size());
    std::unordered_map<int, pcl::PointXYZ> vectors;

    pl_matcher->local_correspondences(key_points, cloud1, cloud2, new_cloud);
    return vectors;
}

std::unordered_map<int, pcl::PointXYZ>
FlowFilter::flow5(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    
    std::vector<int> prev2next(key_points->size());
    std::unordered_map<int, pcl::PointXYZ> vectors;

    pl_matcher_2->local_correspondences(key_points, cloud1, cloud2, new_cloud);
    return vectors;
}

// Create filter 3 to repeat local ICP + Filter
// Testar sobre mesh
std::unordered_map<int, pcl::PointXYZ>
FlowFilter::flow3(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    
    std::vector<int> prev2next(key_points->size());
    std::unordered_map<int, pcl::PointXYZ> vectors;

    std::cout << "C1: " << cloud1->size() << " C2: " << cloud2->size() << std::endl;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::copyPointCloud(*cloud1, *temp);

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr acc_vec (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::copyPointCloud(*cloud1, *acc_vec);

    for (int j=0;j<acc_vec->size(); j++) {
        pcl::PointXYZINormal &acc = acc_vec->points[j];
        acc.normal_x = 0;
        acc.normal_y = 0;
        acc.normal_z = 0;
        acc.intensity = 1;
    }

    for (int k=0; k < patchmatch_levels; k++) {
        vectors.clear();
        std::unordered_map<int, int> matches;

        std::cout << "Flow iter" << std::endl;
        this->flow_neighborhood(temp, cloud2, key_points, vectors, matches);
            
        // Filter to spread correspondences
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs (new pcl::PointCloud<pcl::PointXYZINormal>);
        pcl::copyPointCloud(*temp, *vecs);

        assign_keypoints<pcl::PointXYZINormal, pcl::PointXYZ>(vecs, vectors);

        std::vector<int> valid_keypoints;
        valid_keypoints.push_back(key_points->at(k));
        
        std::cout << "Going to filter " << valid_keypoints.size() << std::endl;
        if (this->descriptor_tech == "PERM") {
            std::cout << "PL Filter" << std::endl;
            perm_filter->filter(temp, vecs, new_cloud);
        } else {
            spatial_filter->pf_3D_vec(temp, vecs, new_cloud, valid_keypoints);
        }
        normalize_colors<pcl::PointXYZINormal>(new_cloud, 0.0001);

        for (int j=0;j<new_cloud->size(); j++) {
            pcl::PointXYZINormal &v = new_cloud->points[j];
            pcl::PointXYZINormal &acc = acc_vec->points[j];
            pcl::PointXYZRGBNormal &pt = temp->points[j];
            pt.x += v.normal_x;
            pt.y += v.normal_y;
            pt.z += v.normal_z;
            acc.normal_x += v.normal_x;
            acc.normal_y += v.normal_y;
            acc.normal_z += v.normal_z;
            v.normal_x = 0;
            v.normal_y = 0;
            v.normal_z = 0;
        }
        //estimate_normal<pcl::PointXYZRGBNormal>(temp, 0.1);
    }
    pcl::copyPointCloud(*acc_vec, *new_cloud);
    
    
    return vectors;
}


std::unordered_map<int, pcl::PointXYZ>
FlowFilter::flow2(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    
    std::vector<int> prev2next(key_points->size());
    std::unordered_map<int, pcl::PointXYZ> vectors;

    std::cout << "C1: " << cloud1->size() << " C2: " << cloud2->size() << std::endl;

    std::unordered_map<int, int> matches;

    // Point matching and CPD
    if (this->match_tech == "FilterReg2") {
        
    } else {
        std::cout << "Flow2 neighborhood" << std::endl;
        // Should sample before and filter later with bilateral filter edge aware?
        this->dense_matcher->local_correspondences(key_points, cloud1, cloud2, new_cloud);
    }
    
    return vectors;
}

std::unordered_map<int, pcl::PointXYZ>
FlowFilter::flow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    
    std::vector<int> prev2next(key_points->size());
    std::unordered_map<int, pcl::PointXYZ> vectors;

    std::cout << "C1: " << cloud1->size() << " C2: " << cloud2->size() << std::endl;

    std::unordered_map<int, int> matches;

    // Point matching and CPD
    if (this->descriptor_tech == "SHOT") {
        this->flow_descriptor(cloud1, cloud2, key_points, vectors, matches);
    }  else if (this->match_tech == "CPD") {
        std::cout << "Flow cpd" << std::endl;
        this->flow_cpd(cloud1, cloud2, key_points, vectors);
    } else if (this->match_tech == "SHOT_RANSAC") {
        std::cout << "SHOT RANSAC" << std::endl;
        this->matcher_SHOT_RANSAC->local_correspondences(key_points, cloud1, cloud2, vectors, matches);
        pcl::copyPointCloud(*cloud1, *new_cloud);
        for (std::pair<int, pcl::PointXYZ> vec: vectors) {
            pcl::PointXYZINormal &p = new_cloud->points[vec.first];
            p.normal_x = vec.second.x;
            p.normal_y = vec.second.y;
            p.normal_z = vec.second.z;
        }
        return vectors;
    } else if (this->descriptor_tech == "FilterReg") {
        std::cout << "FilterReg neighborhood" << std::endl;
        this->matcher_FilterReg->local_correspondences(key_points, cloud1, cloud2, vectors, matches);
    } else {
        std::cout << "Flow neighborhood" << std::endl;
        this->flow_neighborhood(cloud1, cloud2, key_points, vectors, matches);
    }
    
    // Optimization as Dewan
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (this->match_tech == "OPTIMIZATION") {
        std::cout << "running optimization" << std::endl;
        optimizer.optimize(cloud1, cloud2, matches, vectors, new_cloud);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "Elapsed time: " << elapsed_secs << std::endl;
        return vectors;
    } else if (this->match_tech == "PERM") {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs (new pcl::PointCloud<pcl::PointXYZINormal>);
        pcl::copyPointCloud(*cloud1, *vecs);

        assign_keypoints<pcl::PointXYZINormal, pcl::PointXYZ>(vecs, vectors);
        begin = std::chrono::steady_clock::now();
        perm_filter->filter(cloud1, vecs, new_cloud);
        std::cout << "Normalization" << std::endl;
        normalize_colors<pcl::PointXYZINormal>(new_cloud, 0.0001);
        std::cout << "Normalization Done" << std::endl;
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "Elapsed filter time: " << elapsed_secs << std::endl;
                
        return vectors;
    }
    
    // Filter to spread correspondences
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::copyPointCloud(*cloud1, *vecs);

    assign_keypoints<pcl::PointXYZINormal, pcl::PointXYZ>(vecs, vectors);

    std::vector<int> valid_keypoints;
    valid_keypoints.reserve(vectors.size());
    for (int i = 0 ; i < key_points->size(); i++) {
      if (vectors.find(key_points->at(i)) != vectors.end()) {
        valid_keypoints.push_back(key_points->at(i));
      }
    }

    std::cout << "Going to filter " << valid_keypoints.size() << std::endl;
    begin = std::chrono::steady_clock::now();
    spatial_filter->pf_3D_vec(cloud1, vecs, new_cloud, valid_keypoints);
    std::cout << "Going to normalize" << std::endl;
    normalize_colors<pcl::PointXYZINormal>(new_cloud, 0.0001);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed filter time: " << elapsed_secs << std::endl;   


    // begin = std::chrono::steady_clock::now();
    // pcl::BilateralFilter<pcl::PointXYZINormal> b;
    // b.setInputCloud(vecs);
    // b.setHalfSize(0.05);
    // b.setStdDev(0.08);
    // b.filter(*new_cloud);
    // normalize_colors<pcl::PointXYZINormal>(new_cloud, 0.0001);
    // std::cout << "Normalization Done" << std::endl;
    // end = std::chrono::steady_clock::now();
    // elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    // std::cout << "Elapsed filter time: " << elapsed_secs << std::endl;

    return vectors;
}

std::unordered_map<int, pcl::PointXYZ>
FlowFilter::optimize(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
        std::shared_ptr<std::vector<int>> key_points,
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    
    std::vector<int> prev2next(key_points->size());
    std::unordered_map<int, pcl::PointXYZ> vectors;
    std::unordered_map<int, int> matches;

    std::cout << "C1: " << cloud1->size() << " C2: " << cloud2->size() << std::endl;

    if (this->descriptor_tech == "SHOT") {
        this->flow_descriptor(cloud1, cloud2, key_points, vectors, matches);
    }  else if (this->match_tech == "CPD") {
        std::cout << "Flow cpd" << std::endl;
        this->flow_cpd(cloud1, cloud2, key_points, vectors);
    } else {
        std::cout << "Flow neighborhood" << std::endl;
        this->flow_neighborhood(cloud1, cloud2, key_points, vectors, matches);
    }
    optimizer.optimize(cloud1, cloud2, matches, vectors, new_cloud);
    return vectors;
}

std::unordered_map<int, pcl::PointXYZ>
Spread::cloud_flow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    std::cout << "HERE" << std::endl;
    return this->flow_filter->cloud_flow(cloud1, cloud2, new_cloud);
}

std::unordered_map<int, pcl::PointXYZ>
Spread::optimize_flow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {    
    return this->flow_filter->optimize_flow(cloud1, cloud2, new_cloud);
}

std::unordered_map<int, pcl::PointXYZ>
Spread::ioptimize_flow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    std::cout << "MODE: " << mode << std::endl;
    for (int i = 0; i < cloud1->points.size(); i++) {
        cloud1->points[i].r = 175;
        cloud1->points[i].g = 175;
        cloud1->points[i].b = 175;
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

    std::shared_ptr<std::vector<int>> key_points = cI->points;
    
    return this->flow_filter->optimize_flow(cloud1, cloud2, key_points, new_cloud);
}

std::unordered_map<int, pcl::PointXYZ>
Spread::icloud_flow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    std::cout << "MODE: " << mode << std::endl;
    
    for (int i = 0; i < cloud1->points.size(); i++) {
        cloud1->points[i].r = 175;
        cloud1->points[i].g = 175;
        cloud1->points[i].b = 175;
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

    std::shared_ptr<std::vector<int>> key_points = cI->points;
    
    return this->flow_filter->cloud_flow(cloud1, cloud2, key_points, new_cloud);
}

void 
Spread::icloud_color(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    std::cout << "MODE: " << mode << std::endl;
    
    for (int i = 0; i < cloud1->points.size(); i++) {
        cloud1->points[i].r = 175;
        cloud1->points[i].g = 175;
        cloud1->points[i].b = 175;
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

    std::shared_ptr<std::vector<int>> key_points = cI->points;

    this->flow_filter->color(cloud1, cloud2, key_points, new_cloud);
}

void project_points(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud,
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result,
 std::shared_ptr<std::vector<int>> points) {

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZINormal> ());

    if (points) {
        pcl::copyPointCloud (*cloud, *points, *new_cloud);
    } else {
        new_cloud = cloud;
    }
    std::cout << "Cloud to merge: " << new_cloud->size() << " " << result->size() << std::endl;
    for (int i = 0; i < new_cloud->points.size(); i++) {
        pcl::PointXYZINormal p = new_cloud->points[i];
        pcl::PointXYZRGBNormal new_point;
        new_point.x = p.x + p.normal_x;
        new_point.y = p.y + p.normal_y;
        new_point.z = p.z + p.normal_z;
        new_point.r = 0;
        new_point.g = 255;
        new_point.b = 0;
        result->points.push_back(new_point);
    }
    std::cout << "Result cloud: " << result->size() << std::endl;
}

void project_points(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud,
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result) {

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZINormal> ());

    for (int i = 0; i < cloud->points.size(); i++) {
        pcl::PointXYZINormal p = cloud->points[i];
        pcl::PointXYZRGBNormal new_point;
        new_point.x = p.x + p.normal_x;
        new_point.y = p.y + p.normal_y;
        new_point.z = p.z + p.normal_z;
        new_point.r = 255;
        new_point.g = 0;
        new_point.b = 0;
        result->points.push_back(new_point);
    }
    for (int i = 0; i < cloud2->points.size(); i++) {
        pcl::PointXYZRGBNormal p = cloud2->points[i];
        p.r = 0;
        p.g = 0;
        p.b = 255;
        result->points.push_back(p);
    }

    std::cout << "Result2 cloud: " << result->size() << std::endl;
}