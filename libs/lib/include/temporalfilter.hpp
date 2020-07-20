#pragma once 

#include <unordered_map>
#include <vector>

#include "util.hpp"
#include "keypoints.hpp"
#include "pf_vector.hpp"
#include "match.hpp"

typedef std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> point_vector;


class TemporalFilter
{
    public:
    std::shared_ptr<point_vector> pl0s;
    point_vector p_results;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr p_icp;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr p_pos;
    float lmbd, radius, sigma_r_spatial, sigma_s_spatial, sigma_r_flow, sigma_s_flow;

    TemporalFilter(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr initial_cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr initial_flow ,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow , 
    float lmbd, float radius,
    float sigma_r_spatial, float sigma_s_spatial,
    float sigma_r_flow, float sigma_s_flow) {
        
        pl0s = std::make_shared<point_vector>(2);
        for (int i=0; i < pl0s->size(); i++) {
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr l (new pcl::PointCloud<pcl::PointXYZINormal>());
            pl0s->at(i) = l;
            pcl::copyPointCloud(*initial_flow, *pl0s->at(i));
            initialize4d<pcl::PointXYZINormal>(pl0s->at(i), 0);
        }
        p_results = point_vector(2);
        // Has the points/flow for t-1->t
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr f (new pcl::PointCloud<pcl::PointXYZINormal>());
        p_results[0] = f;
        pcl::copyPointCloud(*initial_flow, *p_results[0]);

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr f2 (new pcl::PointCloud<pcl::PointXYZINormal>());
        p_icp = f2;
        pcl::copyPointCloud(*icp_flow, *p_icp);

        // Has the points/normals for t-1
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr l (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        p_pos = l;
        pcl::copyPointCloud(*initial_cloud, *p_pos);
        this->lmbd = lmbd;
        this->radius = radius;
        this->sigma_r_spatial = sigma_r_spatial;
        this->sigma_s_spatial = sigma_s_spatial;
        this->sigma_r_flow = sigma_r_flow;
        this->sigma_s_flow = sigma_s_flow;
    }

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr get_last_flow() {
        return p_results[0];
    }

    void filter_temporal(
    pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud1,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud2,
    pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr flow,
    pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr icp_p);

    void warp_base(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                                pcl::PointCloud<pcl::PointXYZINormal>::Ptr flow,
                                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr warped);

    private:
    float permeability_gradient(const pcl::PointXYZRGBNormal &c2point,
    const pcl::PointXYZRGBNormal &warpedc1,
    const pcl::PointXYZINormal &c2flow,
    const pcl::PointXYZINormal &warpedflow1);

    float permeability_spatial(const pcl::PointXYZRGBNormal &c2point,
    const pcl::PointXYZRGBNormal &warpedc1);

    void warp_spatial(pcl::PointCloud<pcl::PointXYZINormal>::Ptr flow,
                                  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr normals,
                                  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr warped);

    void warp_to_compare(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr warped,
                                     pcl::PointCloud<pcl::PointXYZINormal>::Ptr flow,
                                     pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud1,
                                     pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr normal_to_compare,
                                     pcl::PointCloud<pcl::PointXYZINormal>::Ptr flow_to_compare);
};

