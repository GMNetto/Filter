#pragma once

#include "tracklets.h"
#include "icp.hpp"
#include "util.hpp"
#include "filterPF.hpp"
#include "segment.hpp"

#include <sys/types.h>
#include <dirent.h>
#include <unordered_map>
#include <algorithm>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl-1.8/pcl/common/common_headers.h>
#include <pcl/filters/crop_box.h>

#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/correspondence_rejection_var_trimmed.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_registration.h>

#include "trimmed_icp.h"
#include "patchmatch_icp.hpp"


void read_directory(const std::string& name, std::vector<std::string>& v);

typedef struct feature {
  int id;
  std::string type;
  Eigen::Vector4f centroid;
  Eigen::Vector4f min;
  Eigen::Vector4f max;
} Feature;

struct Line {
    pcl::PointXYZRGBNormal begin, end;

    Line(){}
    Line(pcl::PointXYZRGBNormal begin_, pcl::PointXYZRGBNormal end_): begin(begin_), end(end_){}
};

class KITTI {
    private:
    int current_frame, start, end;
    std::string filepath, cloud_dir, kitti_tech, segment_mode, segmented_dir;
    std::string focus_object;
    std::shared_ptr<Tracklets> tracklets;
    std::unordered_map<int, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> currentObj;

    CustomICP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
    CPD cpd;
    PatchMatchICP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> patchmatch_icp;

    // Trimmed ICP
    pcl::recognition::TrimmedICP<pcl::PointXYZRGBNormal, float> trimmed_icp;

    std::shared_ptr<Spread> filter_match;

    std::vector<std::string> cloud_files;
    float range, uniform_sample;

    pcl::CropBox<pcl::PointXYZRGBNormal> cropFilter;

    std::shared_ptr<Segment> segment;

    std::vector< std::vector< Feature > > features;

    void read_feature_file(std::string &file_name, int idx);

    void get_object(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        Tracklets::tTracklet *t_track, Tracklets::tPose *pose,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr object);

    void apply_motion(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud);

    pcl::PointXYZ populateICPFlow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr original,
                            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed,
                            pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow);

    void transformBox(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow,
                    int tracklet_number,
                    Tracklets::tPose *pose,
                    Tracklets::tTracklet *t_track);

    bool check_focus_object(std::string objectType);

    void setCurrentGT(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud, int frame);

    void transformGT(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                           int frame,
                           pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result,
                           pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow);

    void loadCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud);

    int readKittiVelodyne(std::string &fileName, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud);
    
    int readPCD(std::string &fileName, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud);

    void load_features();

    public:
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr current_cloud;

    KITTI(InputParams &input_params, SegParams &seg_params): icp(input_params), cpd(input_params) {
        pcl::registration::CustomCorrespondenceEstimation<pcl::PointXYZRGBNormal,
        pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr custom_cor(new pcl::registration::CustomCorrespondenceEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(input_params));

        pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointXYZRGBNormal,
         pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr
        cor(new pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>);
        
        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>::Ptr
        rej_sample(new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>);
        rej_sample->setMaximumIterations(input_params.ransac_max_iter);
        rej_sample->setInlierThreshold(input_params.match_icp);

        // pcl::registration::CorrespondenceRejectorMedianDistance::Ptr
        // rej_sample3(new pcl::registration::CorrespondenceRejectorMedianDistance);
        // rej_sample3->setMedianFactor(input_params.patch_min);

        icp = CustomICP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(input_params); 

        icp.setCorrespondenceEstimation(cor);
        icp.addCorrespondenceRejector(rej_sample);

        icp.setMaxCorrespondenceDistance(input_params.match_icp);
        icp.setRANSACOutlierRejectionThreshold(input_params.match_icp);
        std::cout << "Transformation Eps: " << input_params.icp_transformation_eps << std::endl;
        icp.setTransformationEpsilon(input_params.icp_transformation_eps);
        icp.setMaximumIterations(input_params.icp_max_iter);
        icp.setRANSACIterations(input_params.ransac_max_iter);

        patchmatch_icp = PatchMatchICP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(input_params); 
        pcl::registration::PatchMatchCorrespondenceEstimation<pcl::PointXYZRGBNormal,
        pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr pm_cor(new pcl::registration::PatchMatchCorrespondenceEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(input_params));
        patchmatch_icp.setCorrespondenceEstimation(pm_cor);
        patchmatch_icp.setMaxCorrespondenceDistance(input_params.match_icp);
        patchmatch_icp.setRANSACOutlierRejectionThreshold(input_params.match_icp);
        patchmatch_icp.setTransformationEpsilon(input_params.icp_transformation_eps);
        patchmatch_icp.setMaximumIterations(input_params.icp_max_iter);
        patchmatch_icp.setRANSACIterations(input_params.ransac_max_iter);


        filter_match = std::make_shared<Spread>(input_params);
        kitti_tech = input_params.kitti_tech;
        focus_object = input_params.focus_object;
    }

    KITTI(InputParams &input_params, SegParams &seg_params, std::string filepath, std::string cloud_dir, int start, int end, float range=0.2): icp(input_params), cpd(input_params) {
        this->filepath = filepath;
        this->cloud_dir = cloud_dir;
        this->tracklets = std::make_shared<Tracklets>();
        
        this->current_frame = start;
        this->start = start;
        this->end = end;
        this->range = range;

        this->kitti_tech = input_params.kitti_tech;
        this->focus_object = input_params.focus_object;

        segment = std::make_shared<Segment>(seg_params);
        segment_mode = seg_params.segment_mode;
        if (segment_mode == "GT" || segment_mode == "SEGMENT" || segment_mode == "GT_SEG") {
            tracklets->loadFromFile(filepath);
        }

        segmented_dir = seg_params.segmented_dir;

        pcl::registration::CustomCorrespondenceEstimation<pcl::PointXYZRGBNormal,
        pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr custom_cor(new pcl::registration::CustomCorrespondenceEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(input_params));

        pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointXYZRGBNormal,
         pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr
        cor(new pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>);
        
        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>::Ptr
        rej_sample(new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>);
        rej_sample->setMaximumIterations(input_params.ransac_max_iter);
        rej_sample->setInlierThreshold(input_params.match_icp);

        pcl::registration::CorrespondenceRejectorSurfaceNormal::Ptr
        rej_sample2(new pcl::registration::CorrespondenceRejectorSurfaceNormal);
        rej_sample2->setThreshold(input_params.patch_min);

        // pcl::registration::CorrespondenceRejectorMedianDistance::Ptr
        // rej_sample3(new pcl::registration::CorrespondenceRejectorMedianDistance);
        // rej_sample3->setMedianFactor(input_params.patch_min);

        boost::shared_ptr<pcl::registration::CorrespondenceRejectorVarTrimmed >
        rej_sample3(new pcl::registration::CorrespondenceRejectorVarTrimmed );
        rej_sample3->setMinRatio(0.25);
        rej_sample3->setMinRatio(0.75);

        icp = CustomICP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(input_params); 

        icp.setCorrespondenceEstimation(cor);
        icp.addCorrespondenceRejector(rej_sample);
        //icp.addCorrespondenceRejector(rej_sample2);
        //icp.addCorrespondenceRejector(rej_sample3);

        icp.setMaxCorrespondenceDistance(input_params.match_icp);
        icp.setRANSACOutlierRejectionThreshold(input_params.match_icp);
        std::cout << "Transformation Eps: " << input_params.icp_transformation_eps << std::endl;
        icp.setTransformationEpsilon(input_params.icp_transformation_eps);
        icp.setMaximumIterations(input_params.icp_max_iter);
        icp.setRANSACIterations(input_params.ransac_max_iter);

        filter_match = std::make_shared<Spread>(input_params);

        read_directory(cloud_dir, cloud_files);
        std::sort(cloud_files.begin(), cloud_files.end(), compareNat);
        for (int i = start; i < end; i++) {
            std::cout << cloud_files[i] << std::endl;
        }
    }

    //Read and loads first cloud
    void setInitialCloud();

    pcl::PointXYZ icpBox(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr previous,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cropped_cloud2,
    pcl::PointCloud<pcl::PointXYZRGBNormal> &result,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed);

    //Reads and process next cloud
    void nextCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result,
                      pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow);

    void setCurrentObjects(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud, int frame);

    void transformCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                        int frame,
                        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result,
                        pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow);

    void getObjects(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result, int frame);

    void getObjects(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result) {
        this->getObjects(cloud, result, this->current_frame);
    }

    void getObjects(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result);

    void match_filter(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr previous,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cropped_cloud2,
        pcl::PointCloud<pcl::PointXYZRGBNormal> &result,
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow);

    void cpdBox(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr previous,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cropped_cloud2,
        pcl::PointCloud<pcl::PointXYZRGBNormal> &result,
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow);

    void nextFrame() {
        this->current_frame++;
    }

    bool hasNext() {
        return this->current_frame < this->end;
    }
};
