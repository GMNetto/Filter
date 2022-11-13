#include "kitti.hpp"

#include "tracklets.h"
#include "icp.hpp"
#include "util.hpp"
#include "normal.hpp"

#include <ctime>
#include <iostream>
#include <chrono>

#include <boost/filesystem.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl-1.8/pcl/registration/correspondence_rejection_sample_consensus.h>

void read_directory(const std::string& name, std::vector<std::string>& v)
{
    std::cout << "Files dir " << name << std::endl;
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        std::cout << "File dir " << dp->d_name << std::endl;
        v.push_back(dp->d_name);
    }
    closedir(dirp);
}

void KITTI::read_feature_file(std::string &file_name, int idx) {
    std::string line;
    std::ifstream myfile (file_name);
    std::vector<Feature> file_features; 
    if (myfile.is_open())
    {
        while ( std::getline (myfile,line) )
        {
            std::vector<std::string> strs;
            boost::split(strs, line, boost::is_any_of("\t "));
            Feature f;
            f.type = strs[0];
            f.centroid = Eigen::Vector4f(std::stof(strs[1]), std::stof(strs[2]), std::stof(strs[3]), 1);
            f.min = Eigen::Vector4f(std::stof(strs[4]), std::stof(strs[5]), std::stof(strs[6]), 1);
            f.max = Eigen::Vector4f(std::stof(strs[7]), std::stof(strs[8]), std::stof(strs[9]), 1);
            file_features.push_back(f);
        }
        myfile.close();
    }
    features.push_back(file_features);
}

void KITTI::load_features() {
    std::vector<std::string> files;
    read_directory(filepath, files);
    std::sort(files.begin(), files.end(), compareNat);
    features.reserve(files.size());
    for (int i = 0; i < files.size(); i++) {
        std::string path = filepath + files[i];
        read_feature_file(path, i);
    }
}

void KITTI::loadCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud) {
    std::cout << "HERE" << std::endl;
    std::string cloud_file = this->cloud_dir + cloud_files[this->current_frame];
    std::cout << "Reading KITTI: " << cloud_file << std::endl;
    int success = readKittiVelodyne(cloud_file, cloud);
    std::cout << "After KITTI " << success << std::endl;
    if (success < 0) {
        this->readPCD(cloud_file, cloud);
    }
    std::cout << "Loaded cloud" << std::endl;
    estimate_normal<pcl::PointXYZRGBNormal>(cloud, 0.3);
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr groundless (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    segment->ground_removal(cloud, groundless);
    pcl::copyPointCloud(*groundless, *cloud);
    current_cloud = groundless;
    std::cout << "Loaded cloud2" << std::endl;
    if (segment_mode == "GT_SEG") {
        boost::filesystem::create_directory(segmented_dir + std::to_string(this->current_frame));
        std::string final_dir = segmented_dir + std::to_string(this->current_frame) + "/" + "groundless.txt";
        save_txt_file(final_dir, groundless);
    }
}

//Read and loads first cloud
void KITTI::setInitialCloud() {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    loadCloud(cloud1);
    this->setCurrentObjects(current_cloud, this->current_frame);
}

//Reads and process next cloud
void KITTI::nextCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result,
                      pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow){
    this->nextFrame();
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    loadCloud(cloud2);
    std::cout << "NEXT CLOUD " << cloud2->size() << std::endl;
    this->transformCloud(cloud2, this->current_frame, result, icp_flow);
};

pcl::PointXYZ KITTI::populateICPFlow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr original,
                            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed,
                            pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow) {
    pcl::PointXYZ average(0, 0, 0);
    for (int i = 0; i < original->size(); i++) {
        pcl::PointXYZRGBNormal &o_p = original->points[i], &t_p = transformed->points[i]; 
        pcl::PointXYZINormal diff;
        diff.x = o_p.x;
        diff.y = o_p.y;
        diff.z = o_p.z;

        diff.normal_x = t_p.x - o_p.x;
        diff.normal_y = t_p.y - o_p.y;
        diff.normal_z = t_p.z - o_p.z;

        average.x += diff.normal_x;
        average.y += diff.normal_y;
        average.z += diff.normal_z;

        icp_flow->points.push_back(diff);
    }
    average.x /= original->size();
    average.y /= original->size();
    average.z /= original->size();
    return average;
}

void KITTI::apply_motion(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud) {
    float x_m = random_float(-this->range/2, this->range/2), y_m = random_float(0, this->range), z_m = random_float(0, this->range);
    for (int i = 0; i < cloud->size(); i++) {
        pcl::PointXYZRGBNormal &p = cloud->points[i];
        p.x += x_m;
        p.y += y_m;
        p.z += z_m;
    }
}

void KITTI::cpdBox(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr previous,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cropped_cloud2,
        pcl::PointCloud<pcl::PointXYZRGBNormal> &result,
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow) {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    std::cout << "CPD " << previous->size() << " " << cropped_cloud2->size() << std::endl;
    std::clock_t begin = std::clock();
    if (previous->size() == 0 || cropped_cloud2->size() == 0) {
        pcl::copyPointCloud(*previous, *transformed);
    } else {
        cpd.align(previous, cropped_cloud2, transformed);
    }
    std::clock_t end = std::clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time CPD: " << elapsed_secs << std::endl;
    result += (*transformed);
    
    this->populateICPFlow(previous, transformed, icp_flow);
}

pcl::PointXYZ KITTI::icpBox(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr previous,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cropped_cloud2,
        pcl::PointCloud<pcl::PointXYZRGBNormal> &result,
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed) {
    
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    
    std::cout << "transform " << kitti_tech << std::endl;
    if (kitti_tech == "c2ICP") {
        std::cout << "transform" << std::endl;
        icp.setInputSource(previous);
        icp.setInputTarget(cropped_cloud2);
    } else if (kitti_tech == "trICP") {
        trimmed_icp.init(cropped_cloud2);
    } else if (kitti_tech == "pmICP") {
        
        patchmatch_icp.setInputSource(previous);
        patchmatch_icp.setInputTarget(cropped_cloud2);
    }
    
    std::cout << "Match " << previous->size() << " " << cropped_cloud2->size() << std::endl;
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp_transformed (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    
    Eigen::Matrix4f transform;
    if (kitti_tech == "c2ICP") {
        std::cout << "transform" << std::endl;
        icp.align(*transformed);
        transform = icp.getFinalTransformation();
    } else if (kitti_tech == "trICP") {
        transform.setIdentity();
        trimmed_icp.setNewToOldEnergyRatio (0.99f);
        int npo = previous->points.size()/3;
        trimmed_icp.align(*previous, npo, transform);
        pcl::transformPointCloud(*previous, *transformed, transform);
    } else if (kitti_tech == "pmICP") {
        std::cout << "start PM ICP" << std::endl;
        patchmatch_icp.align(*transformed);
        transform = patchmatch_icp.getFinalTransformation();
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed time ICP: " << elapsed_secs << std::endl;

    std::cout << "Affine transform " << transform.matrix() << std::endl;

    result += (*transformed);
    return this->populateICPFlow(previous, transformed, icp_flow);
}

void KITTI::match_filter(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr previous,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cropped_cloud2,
        pcl::PointCloud<pcl::PointXYZRGBNormal> &result,
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow) {

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_flow (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    
    filter_match->cloud_flow(previous, cropped_cloud2, new_flow);
    pcl::copyPointCloud(*previous, *transformed);

    for (int i = 0; i < previous->size(); i++) {
        pcl::PointXYZRGBNormal &o_p = previous->points[i], &t_p = transformed->points[i]; 
        pcl::PointXYZINormal diff = new_flow->points[i];
        
        t_p.x += diff.normal_x;
        t_p.y += diff.normal_y;
        t_p.z += diff.normal_z;

        icp_flow->points.push_back(diff);
    }
    result += (*transformed);
}

void KITTI::get_object(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        Tracklets::tTracklet *t_track, Tracklets::tPose *pose,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr object) {
    Eigen::Vector4f minPoint(-t_track->l / 2.0f, -t_track->w / 2.0, -t_track->h / 2.0, 1.0f);
    Eigen::Vector4f maxPoint( t_track->l / 2.0f,  t_track->w / 2.0,  t_track->h / 2.0, 1.0f);
    Eigen::Vector3f boxTranslation((float) pose->tx, (float) pose->ty, (float) pose->tz + t_track->h / 2.0f);
    Eigen::Vector3f boxRotation((float) pose->rx, (float) pose->ry, (float) pose->rz);

    // Isolates object from cloud 2
    cropFilter.setInputCloud(cloud);
    cropFilter.setMin(minPoint);
    cropFilter.setMax(maxPoint);
    cropFilter.setTranslation(boxTranslation);
    cropFilter.setRotation(boxRotation);
    cropFilter.filter(*object);

    Eigen::Affine3f retrans = pcl::getTransformation(pose->tx, pose->ty,
    pose->tz + t_track->h / 2.0f, (2*M_PI*pose->rx) - M_PI, (2*M_PI*pose->ry) - M_PI, (2*M_PI*pose->rz) - M_PI);

    std::cout << "Affine rot " << cropFilter.getRotation() << std::endl;
    std::cout << "Affine trans " << cropFilter.getTranslation() << std::endl;
    std::cout << "Affine transform " << retrans.matrix() << std::endl;
}


// Applyies ICP between bounding boxes
// Other to apply ICP between bounding box and entire c2 cloud
// Other appllying PatchMatch
void KITTI::transformBox(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                           pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result,
                           pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow,
                           int tracklet_number,
                           Tracklets::tPose *pose,
                           Tracklets::tTracklet *t_track) {
   
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    this->get_object(current_cloud, t_track, pose, temp);

    if (segment_mode == "GT_SEG") {
        std::string final_dir = segmented_dir + std::to_string(this->current_frame) + "/" + std::to_string(tracklet_number) + ".txt";
        save_txt_file(final_dir, temp);
        std::cout << "Including obj2: " << temp->size() << " from " << cloud2->size() << std::endl;
        currentObj[tracklet_number] = temp;
        return;
    }

    auto exists = currentObj.find(tracklet_number);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    if (exists != currentObj.end()) {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr object = currentObj[tracklet_number];
        if (kitti_tech == "bbICP") {
            icpBox(object, temp, *result, icp_flow, transformed);
        } else if (kitti_tech == "c2ICP" || kitti_tech == "pmICP" || kitti_tech == "trICP") {
            pcl::PointXYZ diff = icpBox(object, cloud2, *result, icp_flow, transformed);
        } else if (kitti_tech == "bbCPD") {
            std::cout << "CPD " << kitti_tech << std::endl;
            cpdBox(object, temp, *result, icp_flow);
        } else if (kitti_tech == "matchFilter") {
            match_filter(object, cloud2, *result, icp_flow);
        } else if (kitti_tech == "NONE") {
            *result += *object;
            for (int i = 0; i < object->size(); i++) {
                pcl::PointXYZRGBNormal &o_p = object->points[i]; 
                pcl::PointXYZINormal diff;
                diff.x = o_p.x;
                diff.y = o_p.y;
                diff.z = o_p.z;

                diff.normal_x = 0;
                diff.normal_y = 0;
                diff.normal_z = 0;

                icp_flow->points.push_back(diff);
            }
        }
    }
    std::cout << "Including obj2: " << temp->size() << " from " << cloud2->size() << std::endl;
    currentObj[tracklet_number] = temp;
}

bool KITTI::check_focus_object(std::string objectType) {
    if (focus_object != "All")
        if (focus_object != "Car") {
            if (objectType != focus_object)
                return false;
        } else {
            if (objectType != "Car" && focus_object != "Car")
                return false; 
        }
    return true;
}

void KITTI::transformGT(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                           int frame,
                           pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result,
                           pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow) {
    Tracklets::tPose *pose = new Tracklets::tPose();
    pcl::ModelCoefficients coeffs;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    for (int i = 0; i < tracklets->numberOfTracklets(); i++) {
        if (tracklets->getPose(i, frame, pose)) {
            Tracklets::tTracklet *t_track = tracklets->getTracklet(i);
            if (!this->check_focus_object(t_track->objectType))
                continue;
            // Isolates object
            transformBox(cloud2, result, icp_flow, i, pose, t_track);
        }
    }
}

void KITTI::transformCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                           int frame,
                           pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result,
                           pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow) {
    std::clock_t begin_keypoints = std::clock();
    if (segment_mode == "GT") {
        transformGT(cloud2, frame, result, icp_flow);
    } else if (segment_mode == "GT_SEG") {
        transformGT(cloud2, frame, result, icp_flow);
    }
    std::clock_t end_keypoints = std::clock();
    double elapsed_secs_keypoints = double(end_keypoints - begin_keypoints) / CLOCKS_PER_SEC;
    std::cout << "Elapsed transform cloud time: " << elapsed_secs_keypoints << std::endl;
}

void KITTI::setCurrentGT(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud, int frame) {
    Tracklets::tPose *pose = new Tracklets::tPose();
    std::cout << "Number tracklets: " << tracklets->numberOfTracklets() << std::endl;
    for (int i = 0; i < tracklets->numberOfTracklets(); i++) {
        if (tracklets->getPose(i, frame, pose)) {
            Tracklets::tTracklet *t_track = tracklets->getTracklet(i);
            
            if (!this->check_focus_object(t_track->objectType))
                continue;
                
	        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            this->get_object(cloud, t_track, pose, temp);

            if (temp->size() <= 10) continue;
            if (segment_mode == "GT_SEG") {
                std::string final_dir = segmented_dir + std::to_string(this->current_frame) + "/" + std::to_string(i) + ".txt";
                save_txt_file(final_dir, temp);
            }
            std::cout << "Including obj: " << temp->size() << " from " << cloud->size() << std::endl;
            currentObj[i] = temp;
        }
    }
}

void KITTI::setCurrentObjects(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud, int frame) {
    currentObj.clear();
    if (segment_mode == "GT") {
        setCurrentGT(cloud, frame);
    } else if (segment_mode == "GT_SEG") {
        setCurrentGT(cloud, frame);
    }
}

void KITTI::getObjects(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result) {
    for (auto& it: currentObj) {
        (*result) += (*it.second);
    }
}

void KITTI::getObjects(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result, int frame) {
    Tracklets::tPose *pose = new Tracklets::tPose();
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::ModelCoefficients coeffs;
    pcl::CropBox<pcl::PointXYZRGBNormal> cropFilter;
    for (int i = 0; i < tracklets->numberOfTracklets(); i++) {
        if (tracklets->getPose(i, frame, pose)) {
            Tracklets::tTracklet *t_track = tracklets->getTracklet(i);
            this->get_object(cloud, t_track, pose, temp);
            (*result) += (*temp);
        }
    }
}

int KITTI::readPCD(std::string &fileName, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud) {
    return pcl::io::loadPCDFile(fileName, *cloud);
}

int KITTI::readKittiVelodyne(std::string &fileName, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud) {
    std::ifstream input(fileName, std::ios_base::binary);
    if(!input.good()){
        std::cerr<<"Cannot open file : "<<fileName<<std::endl;
        return -1;
    }

    cloud->clear();
    //cloud->height = 1;

    float trash;
    try {
      for (int i=0; input.good() && !input.eof(); i++) {
        pcl::PointXYZRGBNormal point;
        input.read((char *) &point.x, 3*sizeof(float));
        input.read((char *) &trash, sizeof(float));
        cloud->push_back(point);
      }
      std::cerr<<fileName<<":"<<cloud->size()<<" points"<<std::endl;
      input.close();  
    } catch (...) {
      return -1;
    }
    return 0;
}
