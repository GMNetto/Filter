#include "permutohedral_filter.hpp"


void PermutohedralFilter::filter(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud
    , pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs
    , pcl::PointCloud<pcl::PointXYZINormal>::Ptr result) {
    pcl::copyPointCloud(*vecs, *result);

    std::vector<float> feature, feature2, out, normalize, out_n;

    feature.resize ((vecs->points).size() * 3);
    feature2.resize ((vecs->points).size() * 4);
    normalize.resize ((vecs->points).size() * 1);
    out.resize ((vecs->points).size() * 4);
    out_n.resize ((vecs->points).size() * 1);

    float sigma_s = this->sigma_s;
    float sigma_r = this->sigma_r;
    for (int i=0; i < (cloud->points).size(); i++)
    {
        feature[i*3]=(cloud->points[i].x)/sigma_s;
        feature[i*3+1]=(cloud->points[i].y)/sigma_s;
        feature[i*3+2]=(cloud->points[i].z)/sigma_s;

        // feature[i*6+3]=(cloud->points[i].normal_x)/sigma_r;
        // feature[i*6+4]=(cloud->points[i].normal_y)/sigma_r;
        // feature[i*6+5]=(cloud->points[i].normal_z)/sigma_r;

        feature2[i*4]=(vecs->points[i]).normal_x;
        feature2[i*4+1]=vecs->points[i].normal_y;
        feature2[i*4+2]=vecs->points[i].normal_z;
        feature2[i*4+3]=vecs->points[i].intensity;
        // feature2[i*3]=(object_cloud->points[i].x);
        // feature2[i*3+1]=(object_cloud->points[i].y);
        // feature2[i*3+2]=(object_cloud->points[i].z);
        normalize[i]=1;
    }
    pcl::Permutohedral permutohedral;
    permutohedral.init(feature,3,(vecs->points).size());
    //p.debug();
    std::cout << "Comp Perm" << std::endl;
    permutohedral.compute(out,feature2,4,0,0,(vecs->points).size(),(vecs->points).size());
    std::cout << "Comp Norm Perm" << std::endl;
    permutohedral.compute(out_n,normalize,1,0,0,(vecs->points).size(),(vecs->points).size());

    std::cout << "Norm Perm" << std::endl;
    for (int i = 0; i < vecs->points.size(); i++) {
        pcl::PointXYZINormal &r = result->points[i];
        float den = out_n[i];
        r.normal_x = out[i*4]/den;
        r.normal_y = out[i*4+1]/den;
        r.normal_z = out[i*4+2]/den;
        r.intensity = out[i*4+3]/den;
    }
    std::cout << "End Perm" << std::endl;
}