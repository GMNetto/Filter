#pragma once 

#include <pcl/common/common_headers.h>

#include <vector>
#include "util.hpp"

class SpatialFilter
{
    private:
    int number_initial;
    float radius, sigma_s=2, sigma_r=0.055, sigma_d=1;
    std::vector<int> initial_points, visited, order_visit;
    std::vector<std::vector<int>> neighbors;
    std::vector<std::vector<pcl::PointXYZINormal>> ls;
    std::vector<std::vector<pcl::PointXYZINormal>> rs;
    pcl::PointXYZINormal one;
    float mesh_radius;

    void reset();
    void initialize_s(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud);
    void reset_s();

    void copy_n_vec(std::deque<int>& neighbors, int current_elem);

    void filter_PF_neighbors_vec(int index,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr vecs,
    std::vector<std::vector<pcl::PointXYZINormal>> &s);

    void filter_PF_vec(int index, int neighbor_idx,
    const pcl::PointXYZINormal *current,
    const pcl::PointXYZRGBNormal *current_point,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr vecs,
    std::vector<std::vector<pcl::PointXYZINormal>> &s);

    void BFS_back_vec(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr vecs);

    void BFS_vec(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr vecs,
    int initial_point);

    inline float permeability_radial_vec(const pcl::PointXYZRGBNormal& p, const  pcl::PointXYZRGBNormal& pn,
 const pcl::PointXYZRGBNormal& n, const pcl::PointXYZRGBNormal& nn);

    void get_neighbors(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud);

    public:
    SpatialFilter(InputParams &input_params): SpatialFilter(input_params.number_init,
        input_params.radius_pf, input_params.sigma_s, input_params.sigma_r,
        input_params.sigma_d){
        mesh_radius = input_params.patch_min;
    }

    SpatialFilter(int number_initial,
     float radius, float sigma_s=2, float sigma_r=0.055, float sigma_d=1) {
        this->radius = radius;
        this->sigma_s = sigma_s;
        this->sigma_r = sigma_r;
        this->sigma_d = sigma_d;
        this->number_initial = number_initial;

        one.normal_x = one.normal_y = one.normal_z = 1.f;

        ls = std::vector<std::vector<pcl::PointXYZINormal>>(4);
        rs = std::vector<std::vector<pcl::PointXYZINormal>>(4);
    }

    void pf_3D_vec(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud
    , pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs
    , pcl::PointCloud<pcl::PointXYZINormal>::Ptr result
    , const std::vector<int>& initial_points);

    void set_number_initial(int ni) {
        this->number_initial = std::min(ni, number_initial);
    }
    
};
/*
void pf_3D_vec(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud
, pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs
, pcl::PointCloud<pcl::PointXYZINormal>::Ptr result
, const std::vector<int>& initial_points
, int number_inital, float radius, float sigma_s=100, float sigma_r=0.1, float sigma_d=0.7);

*/