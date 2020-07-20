#pragma once

#include <pcl/common/common_headers.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/filters/extract_indices.h>

#include "util.hpp"

class Segment {
private:
    std::shared_ptr<pcl::ConditionalEuclideanClustering<pcl::PointXYZRGBNormal>> cec;
    std::shared_ptr<pcl::ProgressiveMorphologicalFilter<pcl::PointXYZRGBNormal>> pmf;
    std::shared_ptr<pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBNormal>> sor;
    std::shared_ptr<pcl::SACSegmentation<pcl::PointXYZRGBNormal>> seg;
    std::shared_ptr<pcl::ExtractIndices<pcl::PointXYZRGBNormal>> extract;

    static float cluster_angle, cluster_dist;

    float ground_slope, ground_size, ground_dist;

    static bool customRegionGrowing (const pcl::PointXYZRGBNormal& point_a, const pcl::PointXYZRGBNormal& point_b, float squared_distance);

    void segment_cloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::IndicesClustersPtr clusters);

    void filter_segments(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result);
public:
    Segment(SegParams &params) {
        cec = std::make_shared<pcl::ConditionalEuclideanClustering<pcl::PointXYZRGBNormal>>(true);
        cec->setClusterTolerance (params.cluster_dist);
        cec->setMinClusterSize (params.min_cluster);
        cec->setMaxClusterSize (params.max_cluster);
        cec->setConditionFunction (&Segment::customRegionGrowing);

        std::cout << "ground " << params.ground_size << " " << params.ground_slope << " " << params.ground_dist << std::endl;

        // For an unknown reason, if I initialize it here, it slows the execution down

        // pmf = std::make_shared<pcl::ProgressiveMorphologicalFilter<pcl::PointXYZRGBNormal>>();
        // pmf->setMaxWindowSize (params.ground_size);
        // pmf->setSlope (params.ground_slope);
        // pmf->setInitialDistance (0.1f);
        // pmf->setMaxDistance (params.ground_dist);

        ground_slope = params.ground_slope;
        ground_size = params.ground_size;
        ground_dist = params.ground_dist;

        extract = std::make_shared<pcl::ExtractIndices<pcl::PointXYZRGBNormal>>();
        extract->setNegative (true);

        sor = std::make_shared<pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBNormal>>();
        sor->setMeanK(params.outlier_k);
        sor->setStddevMulThresh(params.outlier_threshold);

        seg = std::make_shared<pcl::SACSegmentation<pcl::PointXYZRGBNormal>>();
        //seg->setOptimizeCoefficients(true);
        seg->setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
        //seg->setModelType(pcl::SACMODEL_PLANE);
        seg->setMethodType(pcl::SAC_RANSAC);
        seg->setEpsAngle((10*3.14)/180);
        seg->setMaxIterations(params.plane_iter);
        seg->setDistanceThreshold(params.plane_dist);

        cluster_angle = params.cluster_angle;
        cluster_dist = params.cluster_dist;

        std::cout << "Final " << std::endl;

    }

    void segment(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result);

    void get_objects(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &result);

    void ground_removal(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result);

    void remove_outliers(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result);

    void remove_planes(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result);

};

void segment_cloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result,
        pcl::IndicesClustersPtr &small,
        pcl::IndicesClustersPtr &large);

void ground_removal(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result);

void flow_segment(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr result,
        pcl::IndicesClustersPtr small,
        pcl::IndicesClustersPtr large);

void down_sample(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result);

void remove_outliers(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result);

void remove_planes(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result);

