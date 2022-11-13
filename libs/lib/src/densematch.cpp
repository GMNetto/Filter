#include "densematch.hpp"

#include "permutohedral.h"

#include <vector>
#include <chrono>
#include <math.h>

void PLMatchTwoWays::get_correspondences(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud2,
                    pcl::Correspondences &correspondences,
                    pcl::Correspondences &correspondences2) {
    std::vector<float> feature, feature2, feature3, out, out2, normalize, normalize2, out_n, out_n2;


    int total_size = cloud1->points.size() + cloud2->points.size(); 
    feature.resize (total_size * 3);
    feature2.resize (total_size * 4);
    feature3.resize (total_size * 4);

    normalize.resize (total_size * 1);
    normalize2.resize (total_size * 1);
    
    out.resize (total_size * 4);
    out2.resize (total_size * 4);

    out_n.resize (total_size * 1);
    out_n2.resize (total_size * 1);    

    int j = 0;
    for (int i=0; i < cloud1->points.size(); i++)
    {
        feature[j*3]=(cloud1->points[i].x)/sigma_s;
        feature[j*3+1]=(cloud1->points[i].y)/sigma_s;
        feature[j*3+2]=(cloud1->points[i].z)/sigma_s;

        feature2[j*4]=0;
        feature2[j*4+1]=0;
        feature2[j*4+2]=0;
        feature2[j*4+3]=0;

        feature3[j*4]=(cloud1->points[i].x);
        feature3[j*4+1]=(cloud1->points[i].y);
        feature3[j*4+2]=(cloud1->points[i].z);
        feature3[j*4+3]=1;

        j++;
    }

    for (int i=0; i < cloud2->points.size(); i++)
    {
        feature[j*3]=(cloud2->points[i].x)/sigma_s;
        feature[j*3+1]=(cloud2->points[i].y)/sigma_s;
        feature[j*3+2]=(cloud2->points[i].z)/sigma_s;

        feature2[j*4]=(cloud2->points[i].x);
        feature2[j*4+1]=(cloud2->points[i].y);
        feature2[j*4+2]=(cloud2->points[i].z);
        feature2[j*4+3]=1;
        
        feature3[j*4]=0;
        feature3[j*4+1]=0;
        feature3[j*4+2]=0;
        feature3[j*4+3]=0;

        j++;

    }  

    std::cout << "Init " << sigma_s << " " << total_size << std::endl;

    std::chrono::steady_clock::time_point begin_init = std::chrono::steady_clock::now();
    p.init(feature,3,total_size);
    std::chrono::steady_clock::time_point end_init = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - begin_init).count();
    std::cout << "Elapsed init time: " << elapsed_secs << std::endl;

    std::cout << "Comp" << std::endl;
    std::chrono::steady_clock::time_point begin_comp = std::chrono::steady_clock::now();

    p.compute(out,feature2,4,0,0,total_size,total_size);

    p.compute(out2,feature3,4,0,0,total_size,total_size);

    std::chrono::steady_clock::time_point end_comp = std::chrono::steady_clock::now();
    elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_comp - begin_comp).count();
    std::cout << "Elapsed comp time: " << elapsed_secs << std::endl;
    
    std::cout << "End" << std::endl;

    pcl::copyPointCloud(*cloud1, *trans_cloud1);
    pcl::copyPointCloud(*cloud2, *trans_cloud2);

    //#pragma omp parallel for
    for (int i = 0; i < trans_cloud1->points.size(); i++) {
        float den = out[i*4+3] + 0.0000001;
        //float den = out_n[i] + 0.0000001;

        pcl::PointXYZRGBNormal &r = trans_cloud1->points[i];
        float c1_x = r.x, c1_y = r.y, c1_z = r.z;

        r.x = (out[i*4]/den);
        r.y = (out[i*4+1]/den);
        r.z = (out[i*4+2]/den);

        r.normal_x = r.x - c1_x;
        r.normal_y = r.y - c1_y;
        r.normal_z = r.z - c1_z;

        correspondences[i].index_query = i;
        correspondences[i].index_match = i;
        correspondences[i].distance = r.normal_x*r.normal_x + r.normal_y*r.normal_y + r.normal_z*r.normal_z;
        
    }

    int to_sum = trans_cloud1->points.size();
    //#pragma omp parallel for
    for (int i = 0; i < trans_cloud2->points.size(); i++) {
        int idx = to_sum + i;
        float den = out2[idx*4+3] + 0.0000001;

        pcl::PointXYZRGBNormal &r = trans_cloud2->points[i];
        float c1_x = r.x, c1_y = r.y, c1_z = r.z;

        r.x = (out2[idx*4]/den);
        r.y = (out2[idx*4+1]/den);
        r.z = (out2[idx*4+2]/den);

        r.normal_x = r.x - c1_x;
        r.normal_y = r.y - c1_y;
        r.normal_z = r.z - c1_z;

        correspondences2[i].index_query = i;
        correspondences2[i].index_match = i;
        correspondences2[i].distance = r.normal_x*r.normal_x + r.normal_y*r.normal_y + r.normal_z*r.normal_z;
        
    }
    std::cout << "Assigned vectors" << std::endl;
}

void PLMatchTwoWays::filter(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs) {
    
    std::vector<float> feature, feature2, out, normalize, out_n;

    feature.resize ((vecs->points).size() * 3);
    feature2.resize ((vecs->points).size() * 4);
    normalize.resize ((vecs->points).size() * 1);
    out.resize ((vecs->points).size() * 4);
    out_n.resize ((vecs->points).size() * 1);

    for (int i=0; i < (cloud->points).size(); i++)
    {
        feature[i*3]=(cloud->points[i].x)/sigma_r;
        feature[i*3+1]=(cloud->points[i].y)/sigma_r;
        feature[i*3+2]=(cloud->points[i].z)/sigma_r;

        feature2[i*4]=(vecs->points[i]).normal_x;
        feature2[i*4+1]=vecs->points[i].normal_y;
        feature2[i*4+2]=vecs->points[i].normal_z;
        feature2[i*4+3]=vecs->points[i].intensity;

        normalize[i]=1;
    }
    p.init(feature,3,(vecs->points).size());
    
    std::cout << "Comp Perm" << std::endl;
    p.compute(out,feature2,4,0,0,(vecs->points).size(),(vecs->points).size());
    std::cout << "Comp Norm Perm" << std::endl;
    p.compute(out_n,normalize,1,0,0,(vecs->points).size(),(vecs->points).size());

    std::cout << "Norm Perm" << std::endl;
    for (int i = 0; i < vecs->points.size(); i++) {
        pcl::PointXYZINormal &vec_r = vecs->points[i];
        float den = out_n[i];
        vec_r.normal_x = out[i*4]/den;
        vec_r.normal_y = out[i*4+1]/den;
        vec_r.normal_z = out[i*4+2]/den;
        vec_r.intensity = out[i*4+3]/den;
    }
    std::cout << "End Perm" << std::endl;
}

float PLMatchTwoWays::copyCorresp (pcl::Correspondences &correspondences, std::vector<int> &indices, pcl::Correspondences &tmp,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src) {
    float total_dist = 0;
    for (int i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        pcl::Correspondence &corresp = correspondences[idx];

        // PL tend to have issues in some points, these are ignored
        if (corresp.distance > match_radius) {
            //std::cout << "Dist: " << corresp.distance << std::endl;
            continue;
        }

        pcl::Correspondence c;
        c.index_query = i;
        c.index_match = corresp.index_match;
        c.distance = corresp.distance;
        total_dist += corresp.distance;

        tmp.push_back(c);
    }
    std::cout << "Tmp corresp size: " << tmp.size() << std::endl;
    return total_dist/tmp.size();
}

void PLMatchTwoWays::transform_keypoint(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src,
    int keypoint_idx,
    Eigen::Matrix4f &transformation,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::transformPointCloud(*src, *transformed, transformation);

    pcl::PointXYZRGBNormal &transformed_point = transformed->points[0];

    #pragma omp critical
    {
        pcl::PointXYZINormal &np = new_cloud->points[keypoint_idx];
        np.normal_x = transformed_point.x - np.x;
        np.normal_y = transformed_point.y - np.y;
        np.normal_z = transformed_point.z - np.z;
        np.intensity = 1;

        //std::cout << "Vector: " << np.normal_x << " " << np.normal_y << " " << np.normal_z << std::endl;
    }
}

float PLMatchTwoWays::get_local_correspondences(pcl::PointXYZRGBNormal &keypoint,
    pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> &tree,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
    pcl::Correspondences &correspondences,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr local_cloud,
    pcl::Correspondences &local_corresp) {
    
    std::vector<int> indices;
    std::vector<float> squared_distances;
    tree.radiusSearch(keypoint, this->patch_radius, indices, squared_distances);

    pcl::copyPointCloud(*cloud, indices, *local_cloud);
            
    local_corresp.reserve(local_cloud->size());
    return copyCorresp(correspondences, indices, local_corresp, cloud, local_cloud);
}

float PLMatchTwoWays::get_transformation_cost(
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
    Eigen::Matrix4f &transformation,
    pcl::Correspondences &correspondences) {

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::transformPointCloud(*src, *transformed, transformation);
    save_txt_file("transformed.txt", transformed);
    float total_dist = 0;
    for (int i = 0; i < correspondences.size(); i++) {
        pcl::Correspondence &corresp = correspondences[i];
        pcl::PointXYZRGBNormal &p = src->points[corresp.index_query], &q = cloud2->points[corresp.index_match];
        total_dist += distance<pcl::PointXYZRGBNormal>(p, q);
    }
    return total_dist/correspondences.size();
}

void PLMatchTwoWays::transform_cloud(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud1,
                    pcl::Correspondences &correspondences,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud2,
                    pcl::Correspondences &correspondences2,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {

    pcl::copyPointCloud(*cloud1, *new_cloud);
    for (int i = 0; i < new_cloud->size(); i++) {
        pcl::PointXYZINormal &np = new_cloud->points[i];

        np.normal_x = 0;
        np.normal_y = 0;
        np.normal_z = 0;
        np.intensity = 0;
    }

    tree.setInputCloud(cloud1);
    tree2.setInputCloud(trans_cloud2);

    #pragma omp parallel for
    for (int i = 0; i < keypoints->size(); ++i)
    {
        int keypoint_idx = keypoints->at(i);
        pcl::PointXYZRGBNormal &keypoint = cloud1->points[keypoint_idx];
        std::vector<int> indices;
        std::vector<float> squared_distances;

        pcl::Correspondences tmp_corresp;
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        float avg_dist_1 = get_local_correspondences(keypoint, tree, cloud1, correspondences, src, tmp_corresp);

        pcl::Correspondences tmp_corresp2;
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src2(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        float avg_dist_2 = get_local_correspondences(keypoint, tree2, trans_cloud2, correspondences2, src2, tmp_corresp2);
        
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr kp_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        kp_cloud->points.push_back(cloud1->points[keypoint_idx]);

        Eigen::Matrix4f transformation, transformation2, transformation_f = Eigen::Matrix4f::Identity();

        if (tmp_corresp.size() != 0 && tmp_corresp2.size() != 0) {
            save_txt_file("src.txt", src);
            estimation.estimateRigidTransformation(*src, *trans_cloud1, tmp_corresp, transformation);
            float cost = get_transformation_cost(src, trans_cloud1, transformation, tmp_corresp);

            save_txt_file("src2.txt", src2);
            estimation.estimateRigidTransformation(*src2, *cloud2, tmp_corresp2, transformation2);
            float cost2 = get_transformation_cost(src2, cloud2, transformation2, tmp_corresp2);

            std::cout << "avg: " << cost << " " << cost2 << std::endl;
            if (cost < cost2) {
                std::cout << "cost1" << std::endl;
                transformation_f = transformation;
            } else {
                std::cout << "cost2" << std::endl;
                transformation_f = transformation2;
            }
            
        } else if (tmp_corresp2.size() == 0) {
            estimation.estimateRigidTransformation(*src, *trans_cloud1, tmp_corresp, transformation);
            transformation_f = transformation;
        } else if (tmp_corresp.size() == 0) {
            estimation.estimateRigidTransformation(*src2, *cloud2, tmp_corresp2, transformation2);
            transformation_f = transformation2;
        }
        
        // Try to transform twice and compare
        transform_keypoint(kp_cloud, keypoint_idx, transformation_f, new_cloud);
        
    }
}

void PLMatchTwoWays::local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    
    std::cout << "PL Two ways" << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    pcl::Correspondences correspondences (cloud1->size());
    pcl::Correspondences correspondences2 (cloud2->size());

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::copyPointCloud(*cloud1, *temp);
    pcl::copyPointCloud(*cloud1, *new_cloud);

    for (int i = 0; i < new_cloud->size(); i++) {
        pcl::PointXYZINormal &np = new_cloud->points[i];

        np.normal_x = 0;
        np.normal_y = 0;
        np.normal_z = 0;
        np.intensity = 1;
    }

    for (int i=0; i < patch_match_iter; i++) {
        
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr acc_vec (new pcl::PointCloud<pcl::PointXYZINormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        
        get_correspondences(temp, cloud2, trans_cloud1, trans_cloud2, correspondences, correspondences2);

        save_txt_file("trans_cloud1.txt", trans_cloud1);
        save_txt_file("trans_cloud2.txt", trans_cloud2);

        transform_cloud(keypoints, temp, trans_cloud1, correspondences, cloud2, trans_cloud2, correspondences2, acc_vec);

        filter(temp, acc_vec);
        normalize_colors<pcl::PointXYZINormal>(acc_vec, 0.0001);

        #pragma omp parallel for
        for (int j = 0; j < acc_vec->size(); j++) {
            pcl::PointXYZINormal &acc = acc_vec->points[j], &np = new_cloud->points[j];
            pcl::PointXYZRGBNormal &pt = temp->points[j];

            np.normal_x += acc.normal_x;
            np.normal_y += acc.normal_y;
            np.normal_z += acc.normal_z;

            pt.x += acc.normal_x;
            pt.y += acc.normal_y;
            pt.z += acc.normal_z;
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed time: " << elapsed_secs << std::endl;
}

/**
Simple PL
**/

void PLMatchSimple::get_new_pos(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud1,
                    std::vector<float> &weights) {
    std::vector<float> feature, feature2, out, normalize, out_n;


    int total_size = cloud1->points.size() + cloud2->points.size(); 
    feature.resize (total_size * 3);
    feature2.resize (total_size * 3);
    normalize.resize (total_size * 1);
    out.resize (total_size * 3);
    out_n.resize (total_size * 1);

    int j = 0;
    for (int i=0; i < cloud1->points.size(); i++)
    {
        feature[j*3]=(cloud1->points[i].x)/sigma_s;
        feature[j*3+1]=(cloud1->points[i].y)/sigma_s;
        feature[j*3+2]=(cloud1->points[i].z)/sigma_s;

        feature2[j*3]=0;
        feature2[j*3+1]=0;
        feature2[j*3+2]=0;

        normalize[j]=0;
        j++;
    }

    for (int i=0; i < cloud2->points.size(); i++)
    {
        feature[j*3]=(cloud2->points[i].x)/sigma_s;
        feature[j*3+1]=(cloud2->points[i].y)/sigma_s;
        feature[j*3+2]=(cloud2->points[i].z)/sigma_s;

        feature2[j*3]=(cloud2->points[i].x);
        feature2[j*3+1]=(cloud2->points[i].y);
        feature2[j*3+2]=(cloud2->points[i].z);
        
        normalize[j]=1;
        j++;

    }

    Eigen::Matrix<float, 3, Eigen::Dynamic> src(3, cloud1->size() + cloud2->points.size());
    j = 0;
    for (size_t i = 0; i < cloud1->size(); ++i) {
        src.col(j) << cloud1->points[i].x/sigma_s, cloud1->points[i].y/sigma_s, cloud1->points[i].z/sigma_s;
        j++;
    }
    for (size_t i = 0; i < cloud1->size(); ++i) {
        src.col(j) << cloud2->points[i].x/sigma_s, cloud2->points[i].y/sigma_s, cloud2->points[i].z/sigma_s;
        j++;
    }

    std::chrono::steady_clock::time_point begin_init_new = std::chrono::steady_clock::now();
    Permutohedral perm;
    perm.init(src);
    std::chrono::steady_clock::time_point end_init_new = std::chrono::steady_clock::now();
    double elapsed_secs_init = std::chrono::duration_cast<std::chrono::milliseconds>(end_init_new - begin_init_new).count();
    std::cout << "Elapsed new init time: " << elapsed_secs_init << std::endl; 

    std::cout << "Init " << sigma_s << " " << total_size << std::endl;

    std::chrono::steady_clock::time_point begin_init = std::chrono::steady_clock::now();
    p.init(feature,3,total_size);
    std::chrono::steady_clock::time_point end_init = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - begin_init).count();
    std::cout << "Elapsed init time: " << elapsed_secs << std::endl;

    std::cout << "Comp" << std::endl;
    std::chrono::steady_clock::time_point begin_comp = std::chrono::steady_clock::now();

    p.compute(out,feature2,3,0,0,total_size,total_size);

    p.compute(out_n,normalize,1,0,0,total_size,total_size);

    std::chrono::steady_clock::time_point end_comp = std::chrono::steady_clock::now();
    elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_comp - begin_comp).count();
    std::cout << "Elapsed comp time: " << elapsed_secs << std::endl;
    
    std::cout << "End" << std::endl;

    pcl::copyPointCloud(*cloud1, *trans_cloud1);

    float w = 0.1; // Noise according to EM
    float m = cloud1->size(), n = cloud2->size(), sigma2 = sigma_s*sigma_s, dim=3;
    float c = (w / (1.0 - w)) * (n / m) * pow(2.0 * sigma2 * M_PI,(dim / 2.0));
    std::cout << "c: " << c << std::endl;
    #pragma omp parallel for
    for (int i = 0; i < trans_cloud1->points.size(); i++) {
        float den = out_n[i];

        pcl::PointXYZRGBNormal &r = trans_cloud1->points[i];
        float c1_x = r.x, c1_y = r.y, c1_z = r.z;

        weights[i] = den/(den + c);

        if (den <= 0.0000001) {
            r.normal_x = 0;
            r.normal_y = 0;
            r.normal_z = 0;
            weights[i] = 0;
            continue;
        }

        //den += 0.0000001;

        r.x = (out[i*3]/den);
        r.y = (out[i*3+1]/den);
        r.z = (out[i*3+2]/den);

        r.normal_x = r.x - c1_x;
        r.normal_y = r.y - c1_y;
        r.normal_z = r.z - c1_z;
    }
    std::cout << "Assigned vectors" << std::endl;
}

void PLMatchSimple::filter(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs) {
    
    std::vector<float> feature, feature2, out, normalize, out_n;

    feature.resize ((vecs->points).size() * 3);
    feature2.resize ((vecs->points).size() * 4);
    normalize.resize ((vecs->points).size() * 1);
    out.resize ((vecs->points).size() * 4);
    out_n.resize ((vecs->points).size() * 1);

    for (int i=0; i < (cloud->points).size(); i++)
    {
        feature[i*3]=(cloud->points[i].x)/sigma_r;
        feature[i*3+1]=(cloud->points[i].y)/sigma_r;
        feature[i*3+2]=(cloud->points[i].z)/sigma_r;

        feature2[i*4]=(vecs->points[i]).normal_x;
        feature2[i*4+1]=vecs->points[i].normal_y;
        feature2[i*4+2]=vecs->points[i].normal_z;
        feature2[i*4+3]=vecs->points[i].intensity;

        normalize[i]=1;
    }

    
    p.init(feature,3,(vecs->points).size());
    
    std::cout << "Comp Perm" << std::endl;
    p.compute(out,feature2,4,0,0,(vecs->points).size(),(vecs->points).size());
    std::cout << "Comp Norm Perm" << std::endl;
    p.compute(out_n,normalize,1,0,0,(vecs->points).size(),(vecs->points).size());

    std::cout << "Norm Perm" << std::endl;
    for (int i = 0; i < vecs->points.size(); i++) {
        pcl::PointXYZINormal &vec_r = vecs->points[i];
        float den = out_n[i];
        vec_r.normal_x = out[i*4]/den;
        vec_r.normal_y = out[i*4+1]/den;
        vec_r.normal_z = out[i*4+2]/den;
        vec_r.intensity = out[i*4+3]/den;
    }
    std::cout << "End Perm" << std::endl;
}

void PLMatchSimple::fill_valid_vectors(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_pos_cloud,
                    std::vector<float> &weights,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {

    pcl::copyPointCloud(*cloud1, *new_cloud);
    
    for (int i=0; i < weights.size(); i++) {
        pcl::PointXYZINormal &np = new_cloud->points[i];
        pcl::PointXYZRGBNormal &npos = new_pos_cloud->points[i];
        float weight = weights[i];

        if (weight != 0) {
            np.normal_x = weight*npos.normal_x;
            np.normal_y = weight*npos.normal_y;
            np.normal_z = weight*npos.normal_z;
            np.intensity = 1;
        } else {
            np.normal_x = 0;
            np.normal_y = 0;
            np.normal_z = 0;
            np.intensity = 0;
        }
        
    }

}

void PLMatchSimple::local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::copyPointCloud(*cloud1, *temp);
    pcl::copyPointCloud(*cloud1, *new_cloud);

    for (int i = 0; i < new_cloud->size(); i++) {
        pcl::PointXYZINormal &np = new_cloud->points[i];

        np.normal_x = 0;
        np.normal_y = 0;
        np.normal_z = 0;
        np.intensity = 1;
    }

    sigma_s = start_sigma_s;

    for (int i=0; i < patch_match_iter; i++) {
        
        std::cout << "new sigma: " << sigma_s << std::endl;

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr acc_vec (new pcl::PointCloud<pcl::PointXYZINormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        std::vector<float> weights(cloud1->size(), 0);

        std::chrono::steady_clock::time_point begin_corresp = std::chrono::steady_clock::now();
        get_new_pos(temp, cloud2, trans_cloud1, weights);
        std::chrono::steady_clock::time_point end_corresp = std::chrono::steady_clock::now();
        double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_corresp - begin_corresp).count();
        std::cout << "Elapsed corresp time: " << elapsed_secs << std::endl;

        fill_valid_vectors(temp, trans_cloud1, weights, acc_vec);
        
        std::chrono::steady_clock::time_point begin_init = std::chrono::steady_clock::now();
        filter(temp, acc_vec);
        std::chrono::steady_clock::time_point end_init = std::chrono::steady_clock::now();
        double elapsed_secs2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - begin_init).count();
        std::cout << "Elapsed init time 2: " << elapsed_secs2 << std::endl;
        normalize_colors<pcl::PointXYZINormal>(acc_vec, 0.0001);


        for (int j = 0; j < acc_vec->size(); j++) {
            pcl::PointXYZINormal &acc = acc_vec->points[j], &np = new_cloud->points[j];
            pcl::PointXYZRGBNormal &pt = temp->points[j];

            np.normal_x += acc.normal_x;
            np.normal_y += acc.normal_y;
            np.normal_z += acc.normal_z;

            pt.x += acc.normal_x;
            pt.y += acc.normal_y;
            pt.z += acc.normal_z;
        }
    }

    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed time: " << elapsed_secs << std::endl;


}


/**
One Way PL
**/

void PLMatch::get_correspondences(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud1,
                    pcl::Correspondences &correspondences) {
    std::vector<float> feature, feature2, out, normalize, out_n;


    int total_size = cloud1->points.size() + cloud2->points.size(); 
    feature.resize (total_size * 3);
    feature2.resize (total_size * 3);
    normalize.resize (total_size * 1);
    out.resize (total_size * 3);
    out_n.resize (total_size * 1);

    int j = 0;
    for (int i=0; i < cloud1->points.size(); i++)
    {
        feature[j*3]=(cloud1->points[i].x)/sigma_s;
        feature[j*3+1]=(cloud1->points[i].y)/sigma_s;
        feature[j*3+2]=(cloud1->points[i].z)/sigma_s;

        feature2[j*3]=0;
        feature2[j*3+1]=0;
        feature2[j*3+2]=0;

        normalize[j]=0;
        j++;
    }

    for (int i=0; i < cloud2->points.size(); i++)
    {
        feature[j*3]=(cloud2->points[i].x)/sigma_s;
        feature[j*3+1]=(cloud2->points[i].y)/sigma_s;
        feature[j*3+2]=(cloud2->points[i].z)/sigma_s;

        feature2[j*3]=(cloud2->points[i].x);
        feature2[j*3+1]=(cloud2->points[i].y);
        feature2[j*3+2]=(cloud2->points[i].z);
        
        normalize[j]=1;
        j++;

    }

    std::cout << "Init " << sigma_s << " " << total_size << std::endl;

    std::chrono::steady_clock::time_point begin_init = std::chrono::steady_clock::now();
    p.init(feature,3,total_size);
    std::chrono::steady_clock::time_point end_init = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - begin_init).count();
    std::cout << "Elapsed init time: " << elapsed_secs << std::endl;

    std::cout << "Comp" << std::endl;
    std::chrono::steady_clock::time_point begin_comp = std::chrono::steady_clock::now();

    p.compute(out,feature2,3,0,0,total_size,total_size);

    p.compute(out_n,normalize,1,0,0,total_size,total_size);

    std::chrono::steady_clock::time_point end_comp = std::chrono::steady_clock::now();
    elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_comp - begin_comp).count();
    std::cout << "Elapsed comp time: " << elapsed_secs << std::endl;
    
    std::cout << "End" << std::endl;

    pcl::copyPointCloud(*cloud1, *trans_cloud1);

    #pragma omp parallel for
    for (int i = 0; i < trans_cloud1->points.size(); i++) {
        float den = out_n[i] + 0.0000001;

        pcl::PointXYZRGBNormal &r = trans_cloud1->points[i];
        float c1_x = r.x, c1_y = r.y, c1_z = r.z;

        r.x = (out[i*3]/den);
        r.y = (out[i*3+1]/den);
        r.z = (out[i*3+2]/den);

        r.normal_x = r.x - c1_x;
        r.normal_y = r.y - c1_y;
        r.normal_z = r.z - c1_z;

        correspondences[i].index_query = i;
        correspondences[i].index_match = i;
        correspondences[i].distance = r.normal_x*r.normal_x + r.normal_y*r.normal_y + r.normal_z*r.normal_z;
        
    }
    std::cout << "Assigned vectors" << std::endl;
}

void PLMatch::filter(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs) {
    
    std::vector<float> feature, feature2, out, normalize, out_n;

    feature.resize ((vecs->points).size() * 3);
    feature2.resize ((vecs->points).size() * 4);
    normalize.resize ((vecs->points).size() * 1);
    out.resize ((vecs->points).size() * 4);
    out_n.resize ((vecs->points).size() * 1);

    for (int i=0; i < (cloud->points).size(); i++)
    {
        feature[i*3]=(cloud->points[i].x)/sigma_r;
        feature[i*3+1]=(cloud->points[i].y)/sigma_r;
        feature[i*3+2]=(cloud->points[i].z)/sigma_r;

        feature2[i*4]=(vecs->points[i]).normal_x;
        feature2[i*4+1]=vecs->points[i].normal_y;
        feature2[i*4+2]=vecs->points[i].normal_z;
        feature2[i*4+3]=vecs->points[i].intensity;

        normalize[i]=1;
    }
    p.init(feature,3,(vecs->points).size());
    
    std::cout << "Comp Perm" << std::endl;
    p.compute(out,feature2,4,0,0,(vecs->points).size(),(vecs->points).size());
    std::cout << "Comp Norm Perm" << std::endl;
    p.compute(out_n,normalize,1,0,0,(vecs->points).size(),(vecs->points).size());

    std::cout << "Norm Perm" << std::endl;
    for (int i = 0; i < vecs->points.size(); i++) {
        pcl::PointXYZINormal &vec_r = vecs->points[i];
        float den = out_n[i];
        vec_r.normal_x = out[i*4]/den;
        vec_r.normal_y = out[i*4+1]/den;
        vec_r.normal_z = out[i*4+2]/den;
        vec_r.intensity = out[i*4+3]/den;
    }
    std::cout << "End Perm" << std::endl;
}

void PLMatch::copyCorresp (pcl::Correspondences &correspondences, std::vector<int> &indices, pcl::Correspondences &tmp,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src) {
    for (int i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        pcl::Correspondence &corresp = correspondences[idx];

        // PL tend to have issues in some points, these are ignored
        if (corresp.distance > match_radius) {
            //std::cout << "Dist: " << corresp.distance << std::endl;
            continue;
        }

        pcl::Correspondence c;
        c.index_query = i;
        c.index_match = corresp.index_match;
        c.distance = corresp.distance;

        tmp.push_back(c);
    }
}

void PLMatch::transform_keypoint(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src,
    int keypoint_idx,
    Eigen::Matrix4f &transformation,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::transformPointCloud(*src, *transformed, transformation);

    pcl::PointXYZRGBNormal &transformed_point = transformed->points[0];

    #pragma omp critical
    {
        pcl::PointXYZINormal &np = new_cloud->points[keypoint_idx];
        np.normal_x = transformed_point.x - np.x;
        np.normal_y = transformed_point.y - np.y;
        np.normal_z = transformed_point.z - np.z;
        np.intensity = 1;

        //std::cout << "Vector: " << np.normal_x << " " << np.normal_y << " " << np.normal_z << std::endl;
    }
}

void PLMatch::transform_cloud(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::Correspondences &correspondences,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {

    pcl::copyPointCloud(*cloud1, *new_cloud);
    for (int i = 0; i < new_cloud->size(); i++) {
        pcl::PointXYZINormal &np = new_cloud->points[i];

        np.normal_x = 0;
        np.normal_y = 0;
        np.normal_z = 0;
        np.intensity = 0;
    }

    tree.setInputCloud(cloud1);
    tree2.setInputCloud(cloud2);

    std::chrono::steady_clock::time_point begin_n = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < keypoints->size(); ++i)
    {
        int keypoint_idx = keypoints->at(i);
        pcl::PointXYZRGBNormal &keypoint = cloud1->points[keypoint_idx];
        std::vector<int> indices;
        std::vector<float> squared_distances;
        
        if (tree.radiusSearch(keypoint, this->patch_radius, indices, squared_distances) > 0)
        {
            
            //std::cout << "t1: " << indices.size() << std::endl;
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr kp_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

            pcl::copyPointCloud(*cloud1, indices, *src);
            
            pcl::Correspondences tmp_corresp;
            tmp_corresp.reserve(src->size());
            copyCorresp(correspondences, indices, tmp_corresp, cloud1, src);
            
            kp_cloud->clear();
            kp_cloud->points.push_back(cloud1->points[keypoint_idx]);

            Eigen::Matrix4f transformation;

            //std::cout << "Estimate trans: " << src->size() << " " << tmp_corresp.size() << std::endl;
 
            //std::cout << src->size() << " " << tmp_corresp.size() << " " << cloud2->size() << std::endl;
            if (tmp_corresp.size() == 0) {
                transformation = Eigen::Matrix4f::Identity();
            } else {
                //std::chrono::steady_clock::time_point begin_rigid_trans = std::chrono::steady_clock::now();
                estimation.estimateRigidTransformation(*src, *cloud2, tmp_corresp, transformation);
                //std::chrono::steady_clock::time_point end_rigid_trans = std::chrono::steady_clock::now();
                //double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_rigid_trans - begin_rigid_trans).count();
                //std::cout << "Elapsed time: " << elapsed_secs << std::endl;
            }
            
            transform_keypoint(kp_cloud, keypoint_idx, transformation, new_cloud);
        }
        
    }
    std::chrono::steady_clock::time_point end_n = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_n - begin_n).count();
    std::cout << "Elapsed N time: " << elapsed_secs << std::endl;
}

void PLMatch::local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    pcl::Correspondences correspondences (cloud1->size());

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::copyPointCloud(*cloud1, *temp);
    pcl::copyPointCloud(*cloud1, *new_cloud);

    for (int i = 0; i < new_cloud->size(); i++) {
        pcl::PointXYZINormal &np = new_cloud->points[i];

        np.normal_x = 0;
        np.normal_y = 0;
        np.normal_z = 0;
        np.intensity = 1;
    }

    int third_iter = floor(patch_match_iter/3);
    for (int i=0; i < patch_match_iter; i++) {
        
        if (i > 2) {
            sigma_s = start_sigma_s/3.0;
        }
        // if (i > 5) {
        //     sigma_s = start_sigma_s/3.5;
        // }
        std::cout << "new sigma: " << sigma_s << std::endl;

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr acc_vec (new pcl::PointCloud<pcl::PointXYZINormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr trans_cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        
        std::chrono::steady_clock::time_point begin_corresp = std::chrono::steady_clock::now();
        get_correspondences(temp, cloud2, trans_cloud1, correspondences);
        std::chrono::steady_clock::time_point end_corresp = std::chrono::steady_clock::now();
        double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_corresp - begin_corresp).count();
        std::cout << "Elapsed corresp time: " << elapsed_secs << std::endl;

        std::chrono::steady_clock::time_point begin_trans = std::chrono::steady_clock::now();
        transform_cloud(keypoints, temp, trans_cloud1, correspondences, acc_vec);
        std::chrono::steady_clock::time_point end_trans = std::chrono::steady_clock::now();
        elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_trans - begin_trans).count();
        std::cout << "Elapsed trans time: " << elapsed_secs << std::endl;


        filter(temp, acc_vec);
        normalize_colors<pcl::PointXYZINormal>(acc_vec, 0.0001);

        for (int j = 0; j < acc_vec->size(); j++) {
            pcl::PointXYZINormal &acc = acc_vec->points[j], &np = new_cloud->points[j];
            pcl::PointXYZRGBNormal &pt = temp->points[j];

            np.normal_x += acc.normal_x;
            np.normal_y += acc.normal_y;
            np.normal_z += acc.normal_z;

            pt.x += acc.normal_x;
            pt.y += acc.normal_y;
            pt.z += acc.normal_z;
        }
    }

    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed time: " << elapsed_secs << std::endl;


}


void DenseMatcher::local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    

    if (cloud2->size() == 0 || cloud1->size() == 0) return;
    std::cout << "Dense Match" << std::endl;
    this->tree.setInputCloud(cloud1);
    this->tree2.setInputCloud(cloud2);

    // pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr s_cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    // pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr s_cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    // pcl::UniformSampling<pcl::PointXYZRGBNormal> uniform_detector;
    // uniform_detector.setInputCloud(cloud1);
    // uniform_detector.setRadiusSearch(0.3);
    // uniform_detector.filter (*s_cloud1);
    // uniform_detector.setInputCloud(cloud2);
    // uniform_detector.filter (*s_cloud2);

    std::vector<int> kp(1);
    pcl::copyPointCloud(*cloud1, *new_cloud);

    for (int i=0; i<new_cloud->size(); i++) {
        pcl::PointXYZINormal &np = new_cloud->points[i];
                    
        np.normal_x = 0;
        np.normal_y = 0;
        np.normal_z = 0;
        np.intensity = 0.0001;
    }


    for (int i = 0; i < static_cast<int>(keypoints->size()); ++i)
    {
        int keypoint_idx = keypoints->at(i);
        pcl::PointXYZRGBNormal &keypoint = cloud1->points[keypoint_idx];
        std::vector<int> indices;
        std::vector<float> squared_distances;
        if (tree.radiusSearch(keypoint, this->patch_radius, indices, squared_distances) > 0)
        {

            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr kp_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

            pcl::copyPointCloud(*cloud1, indices, *src);
            
            std::vector<int> indices2;
            std::vector<float> squared_distances2;
            if (tree2.radiusSearch(keypoint, this->match_radius, indices2, squared_distances2) > 0)
            {
                std::cout << "Matching KP " << i << std::endl;
                pcl::copyPointCloud(*cloud2, indices2, *tgt);
                icp.setInputSource(src);
                icp.setInputTarget(tgt);
                
                icp.align(*transformed);

                for (int j=0; j < transformed->size();j++) {
                    int idx = indices[j];
                    pcl::PointXYZRGBNormal &t = transformed->points[j];
                    pcl::PointXYZRGBNormal &p1 = src->points[j];
                    pcl::PointXYZINormal &np = new_cloud->points[idx];
                    
                    np.normal_x += t.x - p1.x;
                    np.normal_y += t.y - p1.y;
                    np.normal_z += t.z - p1.z;
                    np.intensity += 1;
                }            
            }
        }
    }
    for (int i=0; i<new_cloud->size(); i++) {
        pcl::PointXYZINormal &np = new_cloud->points[i];
                    
        np.normal_x /= np.intensity;
        np.normal_y /= np.intensity;
        np.normal_z /= np.intensity;
        np.intensity = 1;
    }
}