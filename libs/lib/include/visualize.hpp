#pragma once

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>


#include "pf_vector.hpp"
#include "util.hpp"
#include "kitti.hpp"

boost::shared_ptr<pcl::visualization::PCLVisualizer> meshVis (const pcl::PolygonMesh& mesh);

void flow_vis_loop(pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

extern pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr vectors_c;

boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud);

class CloudIteraction
{
    public:
    static pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud;
    pcl::search::KdTree<pcl::PointXYZRGBNormal> tree;
    std::shared_ptr<std::vector<int>> points;

    CloudIteraction() {
        this->points = std::make_shared<std::vector<int>>(0);
        tree.setInputCloud (cloud);
    }

    CloudIteraction(std::shared_ptr<std::vector<int>> keypoints) {
        this->points = keypoints;
        tree.setInputCloud (cloud);
    }

    int markPointIndex(int index) {
        this->points->push_back(index);
    }

    void reset_points() {
        this->points->clear();
    }

    int markPoint(pcl::PointXYZRGBNormal &p) {
        std::vector<int> point_neighbors;
        std::vector<float> distances;
        tree.nearestKSearch (p, 1, point_neighbors, distances);
        points->push_back(point_neighbors[0]);
        return point_neighbors[0];
    }

};

class PointIteraction 
{

    public:
    static pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud;
    static pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr color;
    static std::shared_ptr<CloudIteraction> cI;

    
    static void pointPick (const pcl::visualization::PointPickingEvent &event,
                         void* viewer_void);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> interactionCustomizationVis ();
};

class FlowIteraction 
{

    public:
    static pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud;
    static pcl::PointCloud<pcl::PointXYZINormal>::Ptr vectors;
    static boost::mutex updateModelMutex;
    static bool update;
    
    static void update_cloud (const pcl::visualization::KeyboardEvent &event,
                         void* viewer_void);

    static void vectors_pick (const pcl::visualization::PointPickingEvent &event,
                         void* viewer_void);

    static void get_colored_cloud(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_result);
    static void set_camera(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer);
    static void set_camera(pcl::visualization::PCLVisualizer *viewer);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> flowVis
    (pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud1);

};

std::shared_ptr<std::vector<int>>
regionVis(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud,
  SpatialFilter &s_filter);

class RegionInteraction
{
    public:
    static pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud;
    static pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr color;
    static boost::mutex updateModelMutex;
    static bool update;
    static std::shared_ptr<CloudIteraction> cI;
    std::shared_ptr<std::vector<int>> points;

    RegionInteraction() {
        this->points = std::make_shared<std::vector<int>>(0);
    }

    static void color_cloud(SpatialFilter &s_filter
    , pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr colored_result,
    RegionInteraction &instance);

    static void update_cloud(const pcl::visualization::KeyboardEvent &event,
                         void* viewer_void);

    static void point_pick (const pcl::visualization::PointPickingEvent &event,
                         void* viewer_void);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> regionVis();

    int markPointIndex(int index) {
        this->points->push_back(index);
    }  
};