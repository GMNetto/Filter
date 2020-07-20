
#include "temporalfilter.hpp"
#include "util.hpp"
#include "pf_vector.hpp"

#include <unordered_map>

#include <pcl/common/common_headers.h>

float TemporalFilter::permeability_spatial(const pcl::PointXYZRGBNormal &c2point,
    const pcl::PointXYZRGBNormal &warpedc1) {

    float cos_val = dot_product_normal<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(c2point, warpedc1);
    float fdist = acos(std::max(std::min(1.0f, cos_val), -1.0f));

    float denominator = M_PI*sigma_r_spatial;
    float aux = pow(abs(fdist/denominator), sigma_s_spatial);

    pcl::PointXYZ diff(c2point.x - warpedc1.x, c2point.y - warpedc1.y, c2point.z - warpedc1.z);

    float ddist = norm<pcl::PointXYZ>(diff);
    return std::min(1/(1 + aux), 1.0f);
}

float TemporalFilter::permeability_gradient(const pcl::PointXYZRGBNormal &c2point,
    const pcl::PointXYZRGBNormal &warpedc1,
    const pcl::PointXYZINormal &c2flow,
    const pcl::PointXYZINormal &warpedflow1) {

    pcl::PointXYZINormal diff_flow;
    diff_flow.getNormalVector3fMap() = c2flow.getNormalVector3fMap() - warpedflow1.getNormalVector3fMap();
    float fdist = norm_normal<pcl::PointXYZINormal>(diff_flow);

    //std::cout << "Dist: " << fdist  << " sigma r: " << sigma_r_flow << " sigma s: " << sigma_s_flow << std::endl;

    float denominator = sqrt(3)*sigma_r_flow;
    float aux = pow(abs(fdist/denominator), sigma_s_flow);

    
    pcl::PointXYZ diff(c2point.normal_x - warpedc1.normal_x, c2point.normal_y - warpedc1.normal_y, c2point.normal_z - warpedc1.normal_z);

    float ddist = norm<pcl::PointXYZ>(diff);
    return std::min(1/(1 + aux), 1.0f);
}

void TemporalFilter::warp_base(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                                pcl::PointCloud<pcl::PointXYZINormal>::Ptr flow,
                                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr warped) {
    for (int i = 0; i < warped->size(); i++) {
        pcl::PointXYZRGBNormal &p = warped->points[i];
        pcl::PointXYZINormal &f = flow->points[i];
        p.x += f.normal_x;
        p.y += f.normal_y;
        p.z += f.normal_z;
    }
}

// as cloud on t and t-1 have different points, to compare
// I will interpolate the flow and normals from warped t-1 into the 
// positions that exist on t
void TemporalFilter::warp_to_compare(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr warped,
                                     pcl::PointCloud<pcl::PointXYZINormal>::Ptr flow,
                                     pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud1,
                                     pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr normal_to_compare,
                                     pcl::PointCloud<pcl::PointXYZINormal>::Ptr flow_to_compare) {
    
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> tree;
    tree.setInputCloud (warped);

    pcl::Normal delta_normal, delta_flow;
    float sum_ = 0;


    for (int i = 0; i < cloud1->size(); i++) {
        pcl::PointXYZRGBNormal p = cloud1->points[i], &warp_n = normal_to_compare->points[i];
        pcl::PointXYZINormal &warp_f = flow_to_compare->points[i];

        
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        if (tree.radiusSearch (p, 0.2, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
        {
            for (int j = 0; j < pointIdxRadiusSearch.size(); j++) {
                int idx = pointIdxRadiusSearch[j];
                pcl::PointXYZRGBNormal &n = warped->points[idx];
                pcl::PointXYZINormal &n_flow = flow->points[idx];
                float dist_squared = pointRadiusSquaredDistance[j]; 
                float w = exp(dist_squared)*exp(dist_squared);

                delta_normal.getNormalVector3fMap() += w*n.getNormalVector3fMap();
                delta_flow.getNormalVector3fMap() += w*n_flow.getNormalVector3fMap();
                sum_ += w;
            }
        }
        warp_n.getNormalVector3fMap() = delta_normal.getNormalVector3fMap()/sum_;
        warp_f.getNormalVector3fMap() = delta_flow.getNormalVector3fMap()/sum_;
    }
}

// takes the points/normal on t-1 and using points/flow goes to t
void TemporalFilter::warp_spatial(pcl::PointCloud<pcl::PointXYZINormal>::Ptr flow,
                                  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr normals,
                                  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr warped) {

    
    for (int i = 0; i < flow->size(); i++) {
        pcl::PointXYZINormal &f = flow->points[i];
        pcl::PointXYZRGBNormal &w = warped->points[i];
        w.x += f.normal_x; 
        w.y += f.normal_y;
        w.z += f.normal_z;
    }
}

void TemporalFilter::filter_temporal(
    pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud1,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud2,
    pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr flow,
    pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr icp_p) {
    
    std::cout << "Temp filter" << std::endl;
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> tree;
    tree.setInputCloud (cloud1);

    std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> ls(4);
    for (int i=0; i < 3; i+=2) {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr l (new pcl::PointCloud<pcl::PointXYZINormal>());
        ls[i] = l;
        pcl::copyPointCloud(*flow, *ls[i]);
        initialize4d<pcl::PointXYZINormal>(ls[i], 0);
    }

    for (int i=1; i < 4; i+=2) {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr l (new pcl::PointCloud<pcl::PointXYZINormal>());
        ls[i] = l;
        pcl::copyPointCloud(*flow, *ls[i]);
        initialize4d<pcl::PointXYZINormal>(ls[i], 0.000001);
    }

    // pl0s takes the size of previous t, so t-1, it also has to be warped
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr prev_l = pl0s->at(0), prev_l_ = pl0s->at(1);

    for (int i = 0; i < p_pos->size(); i++) {
    //for (int i = 0; i < 1; i++) {    
        // Flow from t-1 warped
        pcl::PointXYZINormal c_prev_flow = p_results[0]->points[i];
        // Point from t-1 warped 
        pcl::PointXYZRGBNormal c_prev = p_pos->points[i];
        // ICP from t-1 warped
        pcl::PointXYZINormal c_prev_icp = p_icp->points[i];

        c_prev.x = c_prev.x + c_prev_flow.normal_x + c_prev_icp.normal_x;
        c_prev.y = c_prev.y + c_prev_flow.normal_y + c_prev_icp.normal_y;
        c_prev.z = c_prev.z + c_prev_flow.normal_z + c_prev_icp.normal_z;

        pcl::PointXYZINormal prev_l0 = prev_l->points[i], prev_l0_ = prev_l_->points[i];

        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        // We interpolate using the neighbors of the warped t-1 on t
        if (tree.radiusSearch (c_prev, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
        {
            //std::cout << i << " " << pointIdxRadiusSearch.size() << std::endl;
            for (int j = 0; j < pointIdxRadiusSearch.size(); j++) {
            //for (int j = 0; j < 1; j++) {
                int idx = pointIdxRadiusSearch[j];
                //std::cout << "Updating result: " << idx << std::endl;
                pcl::PointXYZRGBNormal c = cloud1->points[idx];
                pcl::PointXYZINormal c_flow = flow->points[idx];

                float per_spatial = permeability_spatial(c, c_prev);
                float per_flow = permeability_gradient(c, c_prev, c_flow, c_prev_flow);
                float per = per_spatial*per_flow, sq_per = per*per;

                // std::cout << "Searching :" << c_prev << std::endl;
                // std::cout << "Searching :" << c  << std::endl;

                // std::cout << "Searching :" << c_prev_flow << std::endl;
                // std::cout << "Searching :" << c_flow  << std::endl;

                //std::cout << "Perm spatial: " << per_spatial << " Perm flow: " << per_flow << " | " << per << std::endl;
                //std::cout << c_prev_flow << "->" << c_flow << std::endl;

                //per = 1;
                //sq_per = 1;

                // as ls has the size of t, it matches with idx
                ls[0]->points[idx].getNormalVector3fMap() += sq_per*(prev_l0.getNormalVector3fMap() + c_prev_flow.getNormalVector3fMap());
                sum_multi_scalar4d<pcl::PointXYZINormal>(ls[1]->points[idx], 1, per, ls[1]->points[idx]);

                // std::cout << ls[0]->points[idx] << "\n" << ls[1]->points[idx] << std::endl;

                pcl::Normal one(1.f,1.f,1.f);
                ls[2]->points[idx].getNormalVector3fMap() += sq_per*(prev_l0_.getNormalVector3fMap() + one.getNormalVector3fMap());
                sum_multi_scalar4d<pcl::PointXYZINormal>(ls[3]->points[idx], 1, per, ls[3]->points[idx]);

                // std::cout << ls[2]->points[idx] << "\n" << ls[3]->points[idx] << std::endl;
            }
        }
    }
    //std::cout << "Update result" << std::endl;
    //Calculate filtered flow into p_results and l0 into pl0s(check if it is a copy)
    pcl::Normal one(1.f,1.f,1.f);
    pcl::copyPointCloud(*cloud1, *p_pos);
    pcl::copyPointCloud(*icp_p, *p_icp);
    pcl::copyPointCloud(*flow, *p_results[0]);
    for (int i = 0; i < flow->size(); i++) {
    //for (int i = 0; i < 1; i++) {
        // std::cout << "Updating result: " << i << std::endl;
        pcl::PointXYZINormal &l = ls[0]->points[i], &l_ = ls[2]->points[i];
        divide_normal<pcl::PointXYZINormal>(l, ls[1]->points[i], l);
        divide_normal<pcl::PointXYZINormal>(l_, ls[3]->points[i], l_);
        pcl::PointXYZINormal num, den;

        // std::cout << "l: " << l <<std::endl;
        // std::cout << "l_: " << l_ <<std::endl;

        num.getNormalVector3fMap() = l.getNormalVector3fMap() + flow->points[i].getNormalVector3fMap();
        
        den.getNormalVector3fMap() = l_.getNormalVector3fMap() + one.getNormalVector3fMap();
        
        // std::cout << "num: " << num <<std::endl;
        // std::cout << "den: " << den <<std::endl;

        divide_normal<pcl::PointXYZINormal>(num, den, p_results[0]->points[i]);

        // std::cout << "XY: " << flow->points[i] <<std::endl;
        // std::cout << "XYT: " << p_results[0]->points[i] <<std::endl;
    }
    //pcl::copyPointCloud(*flow, *p_results[0]);
    (*pl0s)[0] =  ls[0];
    (*pl0s)[1] =  ls[2];
    std::cout << "End filter" << std::endl;
}
