#pragma once

#include <fstream>
#include <cmath>

#include <boost/property_tree/ptree.hpp>
#include <pcl/console/parse.h>
#include <pcl/features/shot_omp.h>
#include <pcl/common/common_headers.h>
#include <Eigen/Core>

pcl::PointXYZRGBNormal randomPointParallel(pcl::PointXYZRGBNormal &p, float radius);

pcl::PointXYZRGBNormal rodrigues(pcl::PointXYZRGBNormal &v, pcl::PointXYZRGBNormal &k, float angle);

float random_float(float a, float b);

void join_flows(pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow, pcl::PointCloud<pcl::PointXYZINormal>::Ptr flow);

bool compareNat(const std::string& a, const std::string& b);

struct InputParams {
    float normal_radius, keypoint_radius, keypoint_threshold,
        descriptor_radius, number_init,
        sigma_s, sigma_r, sigma_d, radius_pf, match_radius, match_icp,
        uniform_radius, patch_radius, patch_min, cpd_beta, cpd_lambda, cpd_tolerance, icp_transformation_eps;

    float motion_bb;
    std::string keypoint_method, tech, match_tech, descriptor_tech, kitti_tech;
    int ransac_max_iter, icp_max_iter, patchmatch_levels, cpd_max_iterations;
    bool use_icp, use_GT;
    std::string focus_object;

    InputParams(boost::property_tree::ptree &pt) {
        normal_radius =  pt.get<float>("normal_radius");
        keypoint_radius = pt.get<float>("keypoint_radius");
        keypoint_threshold = pt.get<float>("keypoint_threshold");

        descriptor_tech = pt.get<std::string>("descriptorTECH");
        descriptor_radius = pt.get<float>("descriptor_radius");

        number_init = pt.get<float>("number_init");
        sigma_s = pt.get<float>("sigma_s");
        sigma_r = pt.get<float>("sigma_r");
        sigma_d = pt.get<float>("sigma_d");
        radius_pf = pt.get<float>("radius_pf");
        match_radius = pt.get<float>("match_radius");
        match_icp = pt.get<float>("match_icp");
        ransac_max_iter = pt.get<float>("ransac_max_iter");
        icp_max_iter = pt.get<float>("icp_max_iter");
        icp_transformation_eps = pt.get<float>("icp_transformation_eps");
        uniform_radius = pt.get<float>("uniform_radius");
        
        keypoint_method = pt.get<std::string>("keypoint_method");
        tech = pt.get<std::string>("keypointTECH");
        match_tech = pt.get<std::string>("matchTECH");
        patch_radius = pt.get<float>("patch_radius");
        patch_min = pt.get<float>("patch_min");
        patchmatch_levels = pt.get<int>("patchmatch_levels");

        cpd_beta = pt.get<float>("cpd_beta");
        cpd_lambda = pt.get<float>("cpd_lambda");
        cpd_tolerance = pt.get<float>("cpd_tolerance");
        cpd_max_iterations = pt.get<int>("cpd_max_iterations");

        use_icp = pt.get<bool>("use_icp");
        motion_bb = pt.get<float>("motion_bb");
        use_GT = pt.get<bool>("use_GT");
        kitti_tech = pt.get<std::string>("KITTITECH");
        focus_object = pt.get<std::string>("focus_object");
    }
};

struct SegParams {
    
    float plane_iter, plane_dist, outlier_threshold, ground_slope, ground_dist, ground_size, cluster_dist, cluster_angle;
    int outlier_k, max_cluster, min_cluster;
    std::string segment_mode, segmented_dir;

    SegParams(){}

    SegParams(boost::property_tree::ptree &pt) {
        plane_iter = pt.get<float>("plane_iter");
        plane_dist = pt.get<float>("plane_dist");

        outlier_threshold = pt.get<float>("outlier_threshold");
        outlier_k = pt.get<int>("outlier_k");

        ground_slope = pt.get<float>("ground_slope");
        ground_dist = pt.get<float>("ground_dist");
        ground_size = pt.get<float>("ground_size");

        cluster_dist = pt.get<float>("cluster_dist");
        cluster_angle = pt.get<float>("cluster_angle");
        max_cluster = pt.get<int>("max_cluster");
        min_cluster = pt.get<int>("min_cluster");

        segment_mode = pt.get<std::string>("segment_mode");
        segmented_dir = pt.get<std::string>("segmented_dir");
    }
};

bool load_cloud(std::string &f1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                InputParams &input_params);

bool load_pair(std::string &f1, std::string &f2,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                InputParams &input_params);

template <typename Point>
void translate_cloud(typename pcl::PointCloud<Point>::Ptr cloud, float x, float y, float z) {
    for (int i=0; i < cloud->size(); i++) {
        cloud->points[i].x += x;
        cloud->points[i].y += y;
        cloud->points[i].z += z;
    }
}

bool read2files(int argc, char **argv, std::string &f1, std::string &f2);

void readSequence(int argc, char **argv, std::string &dir, std::string &tracklet, int &start, int &end);

template <typename Point, typename Normal>
float dot_product(Point p, Normal normal) {
    return p.x*normal.normal_x + p.y*normal.normal_y + p.z*normal.normal_z;
}

template <typename Normal1, typename Normal2>
inline float dot_product_normal(const Normal1 &n, const Normal2 &normal) {
    return n.normal_x*normal.normal_x + n.normal_y*normal.normal_y + n.normal_z*normal.normal_z;
}

template <typename Vector>
inline float norm(Vector v) {
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

template <typename PointT> inline float 
distance (const PointT& p1, const PointT& p2)
{
    PointT diff;
    diff.getArray3fMap() = p1.getArray3fMap() - p2.getArray3fMap();
    return norm<PointT>(diff);
}

template <typename Normal>
float norm_normal(Normal n) {
    return sqrt(n.normal_x*n.normal_x + n.normal_y*n.normal_y + n.normal_z*n.normal_z);
}

template <typename Point, typename Normal>
void unifypointnormal(Point p, Normal n, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud) {
    cloud->reserve(p->size());
    for (int i = 0; i < p->size(); i++) {
        cloud->points[i].r = 0;
        cloud->points[i].g = 0;
        cloud->points[i].b = 0;
        cloud->points[i].x = p->points[i].x;
        cloud->points[i].y = p->points[i].y;
        cloud->points[i].z = p->points[i].z;
        cloud->points[i].normal_x = p->points[i].normal_x;
        cloud->points[i].normal_y = p->points[i].normal_y;
        cloud->points[i].normal_z = p->points[i].normal_z;
    }
}

template <typename Color, typename Normal>
void color_normals(typename pcl::PointCloud<Color>::Ptr color, typename pcl::PointCloud<Normal>::Ptr normals) {
  for (int i = 0; i < color->size(); i++) {
    color->points[i].r = int(255 * (normals->points[i].normal_x + 1)/2);
    color->points[i].g = int(255 * (normals->points[i].normal_y + 1)/2);
    color->points[i].b = int(255 * (normals->points[i].normal_z + 1)/2);
    //std::cout << int(255 * (normals->points[i].normal_x + 1)/2) << " " << (normals->points[i].normal_x + 1)/2 << " " << color->points[i].b << std::endl;
  }
}

template <typename Color, typename Normal>
void color_normals3(typename pcl::PointCloud<Color>::Ptr color, typename pcl::PointCloud<Normal>::Ptr normals) {
  for (int i = 0; i < color->size(); i++) {
    color->points[i].r = int(255 * normals->points[i].normal_x);
    color->points[i].g = int(255 * normals->points[i].normal_y);
    color->points[i].b = int(255 * normals->points[i].normal_z);
  }
}

template <typename Normal1, typename Normal2>
void transfer_normals(typename pcl::PointCloud<Normal1>::Ptr src, typename pcl::PointCloud<Normal2>::Ptr dst) {
  for (int i = 0; i < src->size(); i++) {
    dst->points[i].getNormalVector3fMap() = src->points[i].getNormalVector3fMap();
  }
}

template <typename Normal>
inline 
void sum3(Normal& first, Normal& second, Normal& third, Normal& result) {
    result.normal_x = first.normal_x + second.normal_x + third.normal_x;
    result.normal_y = first.normal_y + second.normal_y + third.normal_y;
    result.normal_z = first.normal_z + second.normal_z + third.normal_z;
}

template <typename Normal>
inline 
void sum6(Normal& first, Normal& second, Normal& third, Normal& result) {
    result.rx = first.rx + second.rx + third.rx;
    result.ry = first.ry + second.ry + third.ry;
    result.rz = first.rz + second.rz + third.rz;
    result.tx = first.tx + second.tx + third.tx;
    result.ty = first.ty + second.ty + third.ty;
    result.tz = first.tz + second.tz + third.tz;
}

template<typename Point>
void divide(Point &p1, Point &p2, Point &result) {
    result.getArray3fMap() = p1.getArray3fMap()/p2.getArray3fMap();
}

template<typename Normal>
inline void divide_normal(Normal &n1, Normal &n2, Normal &result) {
    result.normal_x = n1.normal_x/n2.normal_x;
    result.normal_y = n1.normal_y/n2.normal_y;
    result.normal_z = n1.normal_z/n2.normal_z;
}

template<typename Normal>
inline void divide_rt(Normal &n1, Normal &n2, Normal &result) {
    result.rx = n1.rx/n2.rx;
    result.ry = n1.ry/n2.ry;
    result.rz = n1.rz/n2.rz;
    result.tx = n1.tx/n2.tx;
    result.ty = n1.ty/n2.ty;
    result.tz = n1.tz/n2.tz;
}

template<typename Normal>
inline void divide_normal4d(Normal &n1, Normal &n2, Normal &result) {
    result.normal_x = n1.normal_x/n2.normal_x;
    result.normal_y = n1.normal_y/n2.normal_y;
    result.normal_z = n1.normal_z/n2.normal_z;
    result.intensity = n1.intensity/n2.intensity;
}

template<typename Normal>
inline void divide_normal7d(Normal &n1, Normal &n2, Normal &result) {
    result.rx = n1.rx/n2.rx;
    result.ry = n1.ry/n2.ry;
    result.rz = n1.rz/n2.rz;
    result.tx = n1.tx/n2.tx;
    result.ty = n1.ty/n2.ty;
    result.tz = n1.tz/n2.tz;
    result.intensity = n1.intensity/n2.intensity;
}

template<typename Normal>
void sum_multi_scalar(Normal &n1, float val1, float val2, Normal &result) {
    result.normal_x = val1*n1.normal_x + val2;
    result.normal_y = val1*n1.normal_y + val2;
    result.normal_z = val1*n1.normal_z + val2;
}

template<typename Normal>
inline
void sum_multi_scalar4d(Normal &n1, float val1, float val2, Normal &result) {

    result.normal_x = val1*n1.normal_x + val2;
    result.normal_y = val1*n1.normal_y + val2;
    result.normal_z = val1*n1.normal_z + val2;

    //result.getNormalVector3fMap() = val1*n1.getNormalVector3fMap();
    //result.getNormalVector3fMap() += Eigen::Vector3f(val2, val2, val2);
    /*result.normal_x = val1*n1.normal_x + val2;
    result.normal_y = val1*n1.normal_y + val2;
    result.normal_z = val1*n1.normal_z + val2;*/
    result.intensity = val1*n1.intensity + val2;
}

template<typename Normal>
inline
void sum_multi_scalar7d(Normal &n1, float val1, float val2, Normal &result) {

    result.rx = val1*n1.rx + val2;
    result.ry = val1*n1.ry + val2;
    result.rz = val1*n1.rz + val2;
    result.tx = val1*n1.tx + val2;
    result.ty = val1*n1.ty + val2;
    result.tz = val1*n1.tz + val2;
    
    result.intensity = val1*n1.intensity + val2;
}

template<typename Normal>
void sum_multi_scalar2(Normal &n1, float val1, float val2, Normal &result) {
    result.normal_x = val1*(n1.normal_x + val2);
    result.normal_y = val1*(n1.normal_y + val2);
    result.normal_z = val1*(n1.normal_z + val2);
}

template<typename Normal>
inline void sum_multi_normal2(const Normal &n1, const Normal &n2, float val1, Normal &result) {
    result.normal_x += val1*(n1.normal_x + n2.normal_x);
    result.normal_y += val1*(n1.normal_y + n2.normal_y);
    result.normal_z += val1*(n1.normal_z + n2.normal_z);
}

template<typename Normal>
inline void sum_multi_rt(const Normal &n1, const Normal &n2, float val1, Normal &result) {
    result.rx += val1*(n1.rx + n2.rx);
    result.ry += val1*(n1.ry + n2.ry);
    result.rz += val1*(n1.rz + n2.rz);
    result.tx += val1*(n1.tx + n2.tx);
    result.ty += val1*(n1.ty + n2.ty);
    result.tz += val1*(n1.tz + n2.tz);
}

template<typename Normal>
inline void sum_multi_normal(Normal &n1, Normal &n2, float val1, Normal &result) {
    result.normal_x = val1*(n1.normal_x + n2.normal_x);
    result.normal_y = val1*(n1.normal_y + n2.normal_y);
    result.normal_z = val1*(n1.normal_z + n2.normal_z);
}

template<typename Normal, typename Point>
void sum_normal_point(Normal &n, Point &p, Point &result) {
    result.x = p.x + n.normal_x;
    result.y = p.y + n.normal_y;
    result.z = p.z + n.normal_z;
}

template<typename Normal>
inline
void initialize(typename pcl::PointCloud<Normal>::Ptr cloud, float val) {
    //pcl::Normal zeros(val, val, val);
    for (int i = 0; i < cloud->size(); i++) {
        cloud->points[i].normal_x = val;
        cloud->points[i].normal_y = val;
        cloud->points[i].normal_z = val;
    }
}

template<typename Point>
inline
Point centroid(typename pcl::PointCloud<Point>::Ptr cloud) {
    Point centroid;
    centroid.x = 0;
    centroid.y = 0;
    centroid.z = 0;
    for (int i = 0; i < cloud->size(); i++) {
        centroid.x += cloud->points[i].x;
        centroid.y += cloud->points[i].y;
        centroid.z += cloud->points[i].z;
    }
    centroid.x /= cloud->size();
    centroid.y /= cloud->size();
    centroid.z /= cloud->size();
    return centroid;
}

template<typename Normal>
inline
void initialize4d(Normal& n, float val) {
    n.normal_x = n.normal_y = n.normal_z = n.intensity = val;
}

template<typename Normal>
inline
void initialize7d(Normal& n, float val) {
    n.rx = n.ry = n.rz = n.tx = n.ty = n.tz = n.intensity = val;
}

template<typename Normal>
inline
void initialize4d(typename pcl::PointCloud<Normal>::Ptr cloud, float val) {
    //pcl::Normal zeros(val, val, val);
    for (int i = 0; i < cloud->size(); i++) {
        initialize4d(cloud->points[i], val);
    }
}

template <typename Normal>
void normalize_normals(typename pcl::PointCloud<Normal>::Ptr normals) {
    for (int i = 0; i < normals->size(); i++) {
        float norm = norm_normal<Normal>(normals->points[i]);
        normals->points[i].normal_x /= norm;
        normals->points[i].normal_y /= norm;
        normals->points[i].normal_z /= norm;
    }
}

template <typename Color>
void normalize_colors(typename pcl::PointCloud<Color>::Ptr colors, float threshold) {
    for (int i = 0; i < colors->size(); i++) {
        if (colors->points[i].normal_x > threshold ||
            colors->points[i].normal_y > threshold ||
            colors->points[i].normal_z > threshold ||
            colors->points[i].intensity > threshold) {

            colors->points[i].normal_x /= colors->points[i].intensity;
            colors->points[i].normal_y /= colors->points[i].intensity;
            colors->points[i].normal_z /= colors->points[i].intensity;
            
        }
    }
}

template <typename Point>
void normalize_rt(typename pcl::PointCloud<Point>::Ptr colors, float threshold) {
    for (int i = 0; i < colors->size(); i++) {
        if (colors->points[i].rx > threshold ||
            colors->points[i].ry > threshold ||
            colors->points[i].rz > threshold ||
            colors->points[i].tx > threshold ||
            colors->points[i].ty > threshold ||
            colors->points[i].tz > threshold ||
            colors->points[i].intensity > threshold) {

            colors->points[i].rx /= colors->points[i].intensity;
            colors->points[i].ry /= colors->points[i].intensity;
            colors->points[i].rz /= colors->points[i].intensity;
            colors->points[i].tx /= colors->points[i].intensity;
            colors->points[i].ty /= colors->points[i].intensity;
            colors->points[i].tz /= colors->points[i].intensity;
            
        }
    }
}

extern double interpolate( double val, double y0, double x0, double y1, double x1 );
extern double blue_jet( double grayscale );
extern double green_jet( double grayscale );
extern double red_jet( double grayscale );

template<typename NormalRGB, typename RGB>
void color_normals2(typename pcl::PointCloud<NormalRGB>::Ptr normalrgb, typename pcl::PointCloud<RGB>::Ptr rgb) {
    for (int i = 0; i < normalrgb->size(); i++) {
        float inclination = acos(normalrgb->points[i].normal_z);
        float normalized = inclination/M_PI;

        //std::cout << normalized << std::endl;

        rgb->points[i].b = int(255 * red_jet(normalized));
        rgb->points[i].g = int(255 * green_jet(normalized));
        rgb->points[i].r = int(255 * blue_jet(normalized));
    }
}

float l2_SHOT352(const pcl::SHOT352& first, const pcl::SHOT352& second);

float l2_FPFHSignature33(const pcl::FPFHSignature33& first, const pcl::FPFHSignature33& second);

int readKittiVelodyne(std::string &fileName, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud);

void readKittiVelodyneIntensity(std::string &fileName, pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);

pcl::PointXYZ average_vector(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud,
  std::shared_ptr<std::vector<int>> points=NULL);

int save_txt_file(std::string f, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud);

int load_txt_file(std::string &f, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud);

void sample_cloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::vector<int> &sampled, float radius);