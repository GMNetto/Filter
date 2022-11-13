#pragma once 

#include <unordered_map>
#include <vector>

#include "util.hpp"
#include "densematch.hpp"
#include "keypoints.hpp"
#include "pf_vector.hpp"
#include "permutohedral_filter.hpp"
#include "match.hpp"
#include "system.hpp"

#include <pcl/common/common_headers.h>
#include <pcl/filters/voxel_grid.h>

void project_points(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud,
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result,
 std::shared_ptr<std::vector<int>> points=NULL);

void project_points(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud,
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result);

class FlowFilter
{
    private:
    std::shared_ptr<Keypoint> keypoints;
    std::shared_ptr<Matcher<pcl::PointXYZ>> matcher;
    std::shared_ptr<Matcher<pcl::PointXYZ>> matcherCPD;
    std::shared_ptr<Matcher<pcl::PointXYZ>> matcherSHOT;
    std::shared_ptr<Matcher<pcl::PointXYZ>> matcher_SHOT_RANSAC;
    std::shared_ptr<Matcher<pcl::PointXYZ>> matcher_FilterReg;
    std::shared_ptr<SpatialFilter> spatial_filter;
    std::shared_ptr<DenseMatcher> dense_matcher;
    std::shared_ptr<PLMatch> pl_matcher;
    std::shared_ptr<PLMatchSimple> pl_matcher_2;
    std::shared_ptr<PermutohedralFilter> perm_filter;
    Optimizer optimizer;

    float descriptor_radius, uniform_radius, match_radius, patch_min;
    int patchmatch_levels;
    std::string descriptor_tech;
    std::string match_tech;
    bool use_GT;

    void
    flow_neighborhood(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                std::unordered_map<int, pcl::PointXYZ> &vectors,
                std::unordered_map<int, int> &prev2next); 

    void
    flow_cpd(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                std::unordered_map<int, pcl::PointXYZ> &vectors);

    void
    flow_descriptor(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    std::shared_ptr<std::vector<int>> key_points,
                    std::unordered_map<int, pcl::PointXYZ> &vectors,
                    std::unordered_map<int, int> &prev2next);       

    std::unordered_map<int, pcl::PointXYZ>
    flow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    std::unordered_map<int, pcl::PointXYZ>
    flow2(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    std::unordered_map<int, pcl::PointXYZ>
    flow3(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    std::unordered_map<int, pcl::PointXYZ>
    flow4(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);
                
    std::unordered_map<int, pcl::PointXYZ>
    flow5(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    std::unordered_map<int, pcl::PointXYZ>
    optimize(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
            std::shared_ptr<std::vector<int>> key_points,
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    public:
    FlowFilter(InputParams &input_params) : optimizer(input_params) {
        this->descriptor_radius = input_params.descriptor_radius;
        this->descriptor_tech = input_params.descriptor_tech;

        this->keypoints = std::make_shared<Keypoint>(input_params);

        this->matcher = std::make_shared<MatchKey>(input_params);
        this->matcherCPD = std::make_shared<MatchKeyCPD>(input_params);
        this->matcherSHOT = std::make_shared<MatchSHOT>(input_params);
        this->matcher_SHOT_RANSAC = std::make_shared<MatchFeatures>(input_params);
        this->matcher_FilterReg = std::make_shared<MatchFilterReg>(input_params);
        this->dense_matcher = std::make_shared<DenseMatcher>(input_params);
        this->pl_matcher = std::make_shared<PLMatch>(input_params);
        this->pl_matcher_2 = std::make_shared<PLMatchSimple>(input_params);

        this->spatial_filter = std::make_shared<SpatialFilter>(input_params);
        this->perm_filter = std::make_shared<PermutohedralFilter>(input_params);
        
        uniform_radius = input_params.uniform_radius;
        use_GT = input_params.use_GT;
        match_tech = input_params.match_tech;
        match_radius = input_params.match_radius;
        patch_min = input_params.patch_min;
        patchmatch_levels = input_params.patchmatch_levels;
    }

    void
    color(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    std::unordered_map<int, pcl::PointXYZ>
    optimize_flow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
            std::shared_ptr<std::vector<int>> key_points,
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
        return this->optimize(cloud1, cloud2, key_points, new_cloud);
    }

    std::unordered_map<int, pcl::PointXYZ>
    optimize_flow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
        
        std::shared_ptr<std::vector<int>> key_points = keypoints->get_keypoints(cloud1);
        return this->optimize(cloud1, cloud2, key_points, new_cloud);
    }

    std::unordered_map<int, pcl::PointXYZ>
    cloud_flow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
        
        std::shared_ptr<std::vector<int>> key_points = keypoints->get_keypoints(cloud1);
        if (match_tech == "DENSE")
            return this->flow2(cloud1, cloud2, key_points, new_cloud);
        if (match_tech == "ITER")
            return this->flow3(cloud1, cloud2, key_points, new_cloud);
        if (match_tech == "PL")
            return this->flow4(cloud1, cloud2, key_points, new_cloud);
        if (match_tech == "PL_2")
            return this->flow5(cloud1, cloud2, key_points, new_cloud);
        return this->flow(cloud1, cloud2, key_points, new_cloud);
    }

    std::unordered_map<int, pcl::PointXYZ>
    cloud_flow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> key_points,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
        if (match_tech == "DENSE")
            return this->flow2(cloud1, cloud2, key_points, new_cloud);
        if (match_tech == "ITER")
            return this->flow3(cloud1, cloud2, key_points, new_cloud);
        if (match_tech == "PL")
            return this->flow4(cloud1, cloud2, key_points, new_cloud);
        if (match_tech == "PL_2")
            return this->flow5(cloud1, cloud2, key_points, new_cloud);
        return this->flow(cloud1, cloud2, key_points, new_cloud);
    }
};

class Spread
{
    private:
    std::shared_ptr<FlowFilter> flow_filter;
    std::string mode;

    void
    get_vectors_4(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr keypoints,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
    std::unordered_map<int, pcl::PointXYZ> &vectors,
    std::unordered_map<int, int> &prev2next);

    public:
    Spread(InputParams &input_params) {
        this->flow_filter = std::make_shared<FlowFilter>(input_params);
        this->mode = input_params.match_tech;
    }

    std::unordered_map<int, pcl::PointXYZ>
    cloud_flow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    std::unordered_map<int, pcl::PointXYZ>
    icloud_flow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    void
    icloud_color(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    std::unordered_map<int, pcl::PointXYZ>
    optimize_flow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    std::unordered_map<int, pcl::PointXYZ>
    ioptimize_flow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);
};

