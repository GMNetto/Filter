#pragma once

#include <unordered_map>

#include <omp.h>

#include <pcl/common/common_headers.h>
#include <pcl/correspondence.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl-1.8/pcl/registration/registration.h>
#include <pcl-1.8/pcl/registration/correspondence_rejection_sample_consensus.h>

#include <pcl/features/fpfh_omp.h>

#include "icp.hpp"

template<typename V>
class Matcher
{
    protected:
    float match_radius, descriptor_radius, inlier_threshold;
    int max_iter_ransac, n_correspondence;
    Eigen::Matrix4f best_transformation;
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> tree, tree2, tree_;

    public:
    Matcher(float m_radius, float d_radius) {
        this->match_radius = m_radius;
        this->descriptor_radius = d_radius;
        this->max_iter_ransac = 5;
        this->inlier_threshold = m_radius;
        this->n_correspondence = 10;
    }

    virtual void set_source_keypoints(std::shared_ptr<std::vector<int>> keypoints,
                            pcl::PointCloud<pcl::SHOT352>::Ptr descriptors) {};

    virtual void set_target_keypoints(std::shared_ptr<std::vector<int>> keypoints,
                            pcl::PointCloud<pcl::SHOT352>::Ptr descriptors) {};

    virtual void local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    std::unordered_map<int, V> &vectors,
                    std::unordered_map<int, int>& prev2next) = 0;

    virtual void local_correspondences(
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    std::unordered_map<int, V> &vectors,
                    std::unordered_map<int, int>& prev2next){};
};

class MatchKey : public Matcher<pcl::PointXYZ>
{
    private:
    CustomICP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
    //pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
    float patch_radius;

    void set_icp(CustomICP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> &icp);

    public:
    
    MatchKey(InputParams &input_params) : Matcher(input_params.match_radius, input_params.descriptor_radius) , icp(input_params) {
        
        // pcl::registration::CustomCorrespondenceEstimation<pcl::PointXYZRGBNormal,
        // pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr
        // custom_cor(new pcl::registration::CustomCorrespondenceEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(input_params));
        
        pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointXYZRGBNormal,
         pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr
        cor(new pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>);
        
        // pcl::registration::CorrespondenceEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr
        // cor(new pcl::registration::CorrespondenceEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>);
        

        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>::Ptr
        rej_sample(new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>);
        rej_sample->setMaximumIterations(input_params.ransac_max_iter);
        rej_sample->setInlierThreshold(input_params.match_radius);

        icp.setCorrespondenceEstimation(cor);
        //icp.setCorrespondenceEstimation(cor);
        icp.addCorrespondenceRejector(rej_sample);

        patch_radius = input_params.patch_radius;

        icp.setMaxCorrespondenceDistance(input_params.match_radius);
        icp.setRANSACOutlierRejectionThreshold(input_params.match_radius);
        icp.setTransformationEpsilon(0.001);
        icp.setMaximumIterations(input_params.icp_max_iter);
        icp.setRANSACIterations(input_params.ransac_max_iter);
    }

    void local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    std::unordered_map<int, pcl::PointXYZ> &vectors,
                    std::unordered_map<int, int>& prev2next) override;

};

class MatchSHOT : public Matcher<pcl::PointXYZ>
{
    private:
    float patch_radius;
    CustomICP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
    pcl::registration::CorrespondenceEstimation<pcl::SHOT352, pcl::SHOT352> correspondence_estimation;
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> tree_key2;

    pcl::SHOTEstimationOMP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::SHOT352> descr_est;
    pcl::FPFHEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::FPFHSignature33> desc;

    std::shared_ptr<std::vector<int>> src_keypoints, tgt_keypoints;
    pcl::PointCloud<pcl::SHOT352>::Ptr src_descriptors, tgt_descriptors;
    
    pcl::registration::CorrespondenceEstimation<pcl::SHOT352, pcl::SHOT352> cor;
    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal> rej;

    void cloud2_keypoints(std::shared_ptr<std::vector<int>> keypoints,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                                    std::vector<int> &keypoints2);

    void get_vector(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                                int index_src,
                                int index_tgt,
                                std::unordered_map<int, pcl::PointXYZ> &vectors);

    public:
    MatchSHOT(InputParams &input_params) : Matcher(input_params.match_radius, input_params.descriptor_radius), icp(input_params) {
        pcl::registration::CustomCorrespondenceEstimation<pcl::PointXYZRGBNormal,
        pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr
        custom_cor(new pcl::registration::CustomCorrespondenceEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(input_params));
        
        //pcl::registration::CorrespondenceEstimation<pcl::SHOT352, pcl::SHOT352>::Ptr
        //cor(new pcl::registration::CorrespondenceEstimation<pcl::SHOT352, pcl::SHOT352>);

        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>::Ptr
        rej_sample(new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>);
        rej_sample->setMaximumIterations(input_params.ransac_max_iter);
        rej_sample->setInlierThreshold(input_params.match_radius);

        rej.setMaximumIterations(input_params.ransac_max_iter);
        rej.setInlierThreshold(input_params.match_radius);

        icp.setCorrespondenceEstimation(custom_cor);
        icp.addCorrespondenceRejector(rej_sample);

        icp.setMaxCorrespondenceDistance(input_params.match_radius);
        icp.setRANSACOutlierRejectionThreshold(input_params.match_radius);
        icp.setTransformationEpsilon(0.001);
        icp.setMaximumIterations(input_params.icp_max_iter);
        icp.setRANSACIterations(input_params.ransac_max_iter);

        patch_radius = input_params.patch_radius;
        descr_est.setRadiusSearch (input_params.descriptor_radius);
        descr_est.setNumberOfThreads(omp_get_thread_num());
        desc.setRadiusSearch (input_params.descriptor_radius);
    }

    void local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    std::unordered_map<int, pcl::PointXYZ> &vectors,
                    std::unordered_map<int, int>& prev2next) override;
};

class MatchFeatures : public Matcher<pcl::PointXYZ>
{
    private:
    float patch_radius;
    pcl::registration::CorrespondenceEstimation<pcl::SHOT352, pcl::SHOT352> correspondence_estimation;
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> tree_key2;

    pcl::SHOTEstimationOMP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::SHOT352> descr_est;

    std::shared_ptr<std::vector<int>> src_keypoints, tgt_keypoints;
    pcl::PointCloud<pcl::SHOT352>::Ptr src_descriptors, tgt_descriptors;
    
    pcl::registration::CorrespondenceEstimation<pcl::SHOT352, pcl::SHOT352> cor;
    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal> rej;

    void cloud2_keypoints(std::shared_ptr<std::vector<int>> keypoints,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                                    std::vector<int> &keypoints2);

    void get_vector(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                                int index_src,
                                int index_tgt,
                                std::unordered_map<int, pcl::PointXYZ> &vectors);

    public:
    MatchFeatures(InputParams &input_params) : Matcher(input_params.match_icp, input_params.descriptor_radius) {
        
        pcl::registration::CorrespondenceEstimation<pcl::SHOT352, pcl::SHOT352>::Ptr
        cor(new pcl::registration::CorrespondenceEstimation<pcl::SHOT352, pcl::SHOT352>);

        rej.setMaximumIterations(input_params.ransac_max_iter);
        rej.setInlierThreshold(input_params.match_icp);

        patch_radius = input_params.patch_radius;
        descr_est.setRadiusSearch (input_params.descriptor_radius);
        descr_est.setNumberOfThreads(omp_get_thread_num());
    }

    void local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    std::unordered_map<int, pcl::PointXYZ> &vectors,
                    std::unordered_map<int, int>& prev2next) override;

    Eigen::Matrix4f get_best_transformation();
};

class MatchKeyCPD : public Matcher<pcl::PointXYZ>
{
    private:
    CPD cpd;
    float patch_radius;

    public:
    MatchKeyCPD(InputParams &input_params) : Matcher(input_params.match_radius, input_params.descriptor_radius), cpd(input_params) {
        patch_radius = input_params.patch_radius;
    }

    void local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    std::unordered_map<int, pcl::PointXYZ> &vectors,
                    std::unordered_map<int, int>& prev2next) override;

};

