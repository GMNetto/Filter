#include "match.hpp"

#include "util.hpp"
#include "keypoints.hpp"
#include "icp.hpp"
#include <omp.h>

#include <pcl/octree/octree_search.h>
#include <pcl/common/common_headers.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/geometry.h>
#include <pcl/registration/icp.h>
#include <pcl-1.8/pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_registration.h>
#include <pcl/registration/gicp.h>

#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <set>


void MatchKey::set_icp(pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> &local_icp) {
        
        pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointXYZRGBNormal,
         pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr
        cor_local(new pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>);
        
        // cor_local->setKSearch(1000);

        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>::Ptr
        rej_sample_local(new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>);
        rej_sample_local->setMaximumIterations(icp.getRANSACIterations());
        rej_sample_local->setInlierThreshold(ransac_radius);

        // pcl::registration::CorrespondenceRejectorOneToOne::Ptr
        // rej_sample_local2(new pcl::registration::CorrespondenceRejectorOneToOne);
        
        local_icp.setCorrespondenceEstimation(cor_local);
        local_icp.addCorrespondenceRejector(rej_sample_local);
        // local_icp.addCorrespondenceRejector(rej_sample_local2);
        local_icp.setMaxCorrespondenceDistance(icp.getMaxCorrespondenceDistance());
        local_icp.setRANSACOutlierRejectionThreshold(ransac_radius);
        local_icp.setTransformationEpsilon(0.001);
        local_icp.setMaximumIterations(icp.getMaximumIterations());
        local_icp.setRANSACIterations(icp.getRANSACIterations());
}

void MatchFilterReg::local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                                     pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                                     pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                                     std::unordered_map<int, pcl::PointXYZ> &vectors,
                                     std::unordered_map<int, int> &prev2next)
{
    if (cloud2->size() == 0 || cloud1->size() == 0) return;
    std::cout << "Match Key" << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    this->tree.setInputCloud(cloud1);
    this->tree2.setInputCloud(cloud2);

    std::vector<int> kp(1);
    for (int i = 0; i < static_cast<int>(keypoints->size()); ++i)
    {
        //std::cout << "K: " << i << std::endl;
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
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

            indices.push_back(keypoint_idx);
            pcl::copyPointCloud(*cloud1, indices, *src);
            
            
            kp_cloud->clear();
            kp_cloud->points.push_back(cloud1->points[keypoint_idx]);

            std::vector<int> indices2;
            std::vector<float> squared_distances2;
            if (tree2.radiusSearch(keypoint, this->match_radius, indices2, squared_distances2) > 0)
            {
                pcl::copyPointCloud(*cloud2, indices2, *tgt);

                save_txt_file("src.txt", src);
                save_txt_file("tgt.txt", tgt);

                system("python3 /home/gustavo/filter/python/local_filterreg.py --cloud1 src.txt --cloud2 tgt.txt --out transform.txt");

                std::cout << "Executing script" << std::endl;

                std::ifstream file;
                file.open( "transform.txt");
                Eigen::Matrix4f transform;

                for( int i=0; i<4; i++) 
                    for( int j=0; j<4; j++) {
                        float number;
                        file >> number;
                        transform(i, j) = number;
                    }

                file.close();

                std::cout << "Matrix read " << transform << std::endl;

                //std::cout << "Aligned: " << transformed->size() << std::endl;
                //Eigen::Matrix4f transform = local_icp.getFinalTransformation();
                
                pcl::transformPointCloud(*kp_cloud, *transformed, transform);
                pcl::PointXYZRGBNormal src_point = kp_cloud->points[0];
                pcl::PointXYZRGBNormal transformed_point = transformed->points[0];
               
                tree2.nearestKSearch(transformed_point, 1, indices2, squared_distances2);
                prev2next[keypoint_idx] = indices2[0];
                pcl::PointXYZ diff;

                diff.getArray3fMap() = transformed_point.getArray3fMap() - src_point.getArray3fMap();
                float vec_norm = norm<pcl::PointXYZ>(diff);
                if (vec_norm >= this->match_radius) {
                    std::cout << "ERROR: " << diff << std::endl;
                    continue; 
                }
                vectors.insert({keypoint_idx, diff});
            }
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed time KP: " << elapsed_secs << std::endl;
    std::cout << "end KP" << std::endl;
}

void MatchKey::local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                                     pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                                     pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                                     std::unordered_map<int, pcl::PointXYZ> &vectors,
                                     std::unordered_map<int, int> &prev2next)
{
    if (cloud2->size() == 0 || cloud1->size() == 0) return;
    std::cout << "Match Key" << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    this->tree.setInputCloud(cloud1);
    this->tree2.setInputCloud(cloud2);

    std::vector<int> kp(1);
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(keypoints->size()); ++i)
    {
        //std::cout << "K: " << i << std::endl;
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
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

            indices.push_back(keypoint_idx);
            pcl::copyPointCloud(*cloud1, indices, *src);
            
            
            kp_cloud->clear();
            kp_cloud->points.push_back(cloud1->points[keypoint_idx]);

            std::vector<int> indices2;
            std::vector<float> squared_distances2;
            if (tree2.radiusSearch(keypoint, this->match_radius, indices2, squared_distances2) > 0)
            {
                //std::cout << "t2: " << indices2.size() << std::endl;
                pcl::copyPointCloud(*cloud2, indices2, *tgt);

                CustomICP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> local_icp;
                //pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> local_icp;
                // #pragma omp critical
                // {
                //     this->set_icp(icp);
                // }
                // pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointXYZRGBNormal,
                // pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr
                // cor_local(new pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>);
                // local_icp.setCorrespondenceEstimation(cor_local);
                local_icp.setMaxCorrespondenceDistance(icp.getMaxCorrespondenceDistance());
                local_icp.setMaximumIterations(icp.getMaximumIterations());
                local_icp.setEuclideanFitnessEpsilon(icp.getEuclideanFitnessEpsilon());
                //std::cout << "Iter " << local_icp.getMaximumIterations () << " dist: " << local_icp.getMaxCorrespondenceDistance () << std::endl;
                //local_icp.setUseReciprocalCorrespondences(true);
                local_icp.setInputSource(src);
                local_icp.setInputTarget(tgt);
                //icp.setInputSource(src);
                //icp.setInputTarget(tgt);
                // save_txt_file("/home/gustavo/filter/build/src.txt", src);
                // save_txt_file("/home/gustavo/filter/build/tgt.txt", tgt);
                //std::cout << "R m: " << src->size() << " " << tgt->size() << std::endl;
                local_icp.align(*transformed);
            
                //std::cout << "Aligned: " << transformed->size() << std::endl;
                Eigen::Matrix4f transform = local_icp.getFinalTransformation();
                
                pcl::transformPointCloud(*kp_cloud, *transformed, transform);
                pcl::PointXYZRGBNormal src_point = kp_cloud->points[0];
                pcl::PointXYZRGBNormal transformed_point = transformed->points[0];
               
                tree2.nearestKSearch(transformed_point, 1, indices2, squared_distances2);
                #pragma omp critical
                {
                    prev2next[keypoint_idx] = indices2[0];
                }
                pcl::PointXYZ diff;

                diff.getArray3fMap() = transformed_point.getArray3fMap() - src_point.getArray3fMap();
                //std::cout << "Vector: " << diff.x << " " << diff.y << " " << diff.z << std::endl;
                float vec_norm = norm<pcl::PointXYZ>(diff);
                if (vec_norm >= this->match_radius) {
                    std::cout << "ERROR: " << diff << std::endl;
                    continue; 
                }
                #pragma omp critical
                {
                    vectors.insert({keypoint_idx, diff});
                }
            }
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed time KP: " << elapsed_secs << std::endl;
    std::cout << "end KP" << std::endl;
}

void MatchSHOT::cloud2_keypoints(std::shared_ptr<std::vector<int>> keypoints,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                                    std::vector<int> &keypoints2) {

    std::vector<int> indices2;
    std::vector<float> squared_distances2;

    for (int i = 0; i < keypoints->size(); i++) {
        int keypoint = keypoints->at(i);
        pcl::PointXYZRGBNormal p = cloud1->points[keypoint];
        if (tree2.radiusSearch(p, match_radius, indices2, squared_distances2) > 0)
        {
            keypoints2.insert(keypoints2.end(), indices2.begin(), indices2.end());
        }
    }

    std::set<int> s;
    std::size_t size = keypoints2.size();
    for( std::size_t i = 0; i < size; ++i ) s.insert( keypoints2[i] );
    keypoints2.assign( s.begin(), s.end() );
}

void MatchSHOT::local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                                    std::unordered_map<int, pcl::PointXYZ> &vectors,
                                    std::unordered_map<int, int> &prev2next)
{


    //pcl::PointCloud<pcl::SHOT352>::Ptr input1_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr input1_descriptors (new pcl::PointCloud<pcl::FPFHSignature33> ());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input1_keypoints (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    
    
    //Estimating time to get points for second cloud
    tree2.setInputCloud (cloud2);
    std::vector<int> keypoints2;
    std::chrono::steady_clock::time_point begin_kp2 = std::chrono::steady_clock::now();
    std::cout << "SHOT/FPFH" << std::endl;
    cloud2_keypoints(keypoints, cloud1, cloud2, keypoints2);
    std::cout << "SHOT/FPFH2" << std::endl;
    std::chrono::steady_clock::time_point end_kp2 = std::chrono::steady_clock::now();
    double elapsed_secs_kp2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_kp2 - begin_kp2).count();
    std::cout << "Elapsed time KP2: " << elapsed_secs_kp2 << std::endl;
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input2_keypoints (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());

    //pcl::copyPointCloud (*cloud2, keypoints2, *input2_keypoints);
    pcl::copyPointCloud (*cloud2, keypoints2, *input2_keypoints);
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    pcl::copyPointCloud (*cloud1, *keypoints, *input1_keypoints);
    // descr_est.setInputCloud (input1_keypoints);
    // descr_est.setInputNormals (cloud1);
    // descr_est.setSearchSurface (cloud1);
    // descr_est.compute (*input1_descriptors);
    desc.setInputCloud (input1_keypoints);
    desc.setInputNormals (cloud1);
    desc.setSearchSurface (cloud1);
    desc.compute (*input1_descriptors);

    //std::cout << "Descriptors 1: " << input1_descriptors->size() << std::endl;

    tree.setInputCloud (cloud1);

    //pcl::PointCloud<pcl::SHOT352>::Ptr input2_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr input2_descriptors (new pcl::PointCloud<pcl::FPFHSignature33> ());
    
    std::vector<int> indices2;
    std::vector<float> squared_distances2;

    //descr_est.setInputCloud (input2_keypoints);
    desc.setInputCloud (input2_keypoints);

    //pcl::SHOTEstimationOMP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::SHOT352> descr_est;
    //descr_est.setRadiusSearch (descriptor_radius);
    //descr_est.setSearchSurface (cloud2);
    //descr_est.setInputNormals (cloud2);
    desc.setSearchSurface (cloud2);
    desc.setInputNormals (cloud2);

    std::cout << "Calculating descriptors cloud2: " << input2_keypoints->size() << std::endl;

    //descr_est.setInputCloud (cloud2);
    if (input1_descriptors->size () == 0)
        return;

    std::chrono::steady_clock::time_point begin_desc = std::chrono::steady_clock::now();
    //descr_est.compute (*input2_descriptors);
    desc.compute (*input2_descriptors);
    std::chrono::steady_clock::time_point end_desc = std::chrono::steady_clock::now();
    double elapsed_secs_desc = std::chrono::duration_cast<std::chrono::milliseconds>(end_desc - begin_desc).count();
    std::cout << "Elapsed time DESC: " << elapsed_secs_desc << std::endl;

    pcl::CorrespondencesPtr temp_correspondences(new pcl::Correspondences);
    pcl::Correspondences correspondences;
    
    //pcl::KdTreeFLANN<pcl::SHOT352> match_search;
    pcl::KdTreeFLANN<pcl::FPFHSignature33> match_search;
    match_search.setInputCloud (input2_descriptors);
    for (size_t i = 0; i < input1_descriptors->size (); ++i)
    {
        std::vector<int> neigh_indices (1);
        std::vector<float> neigh_sqr_dists (1);
        // if (!std::isfinite (input1_descriptors->at (i).descriptor[0])) //skipping NaNs
        // {
        //     continue;
        // }
        int found_neighs = match_search.nearestKSearch (input1_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
        
        int idx1 = keypoints->at(i), idx2 = neigh_indices[0];

        pcl::PointXYZRGBNormal p = input1_keypoints->points[i], q = input2_keypoints->points[idx2];
        pcl::PointXYZ diff(p.x - q.x, p.y - q.y, p.z - q.z);
        float ddist = norm<pcl::PointXYZ>(diff);
        //std::cout << "KP " << i << " " << "dist " << ddist << " Feature dist " << neigh_sqr_dists[0] << std::endl;
        if(ddist < match_radius) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
        {
            pcl::Correspondence corr (i, neigh_indices[0], ddist);
            temp_correspondences->push_back(corr);
        }
    }

    
    std::cout << "Correspondences Calc: " << temp_correspondences->size() << std::endl;
    // Go over correspondences and remove points distance > T
    // Use rejection RANSAC
    pcl::PCLPointCloud2::Ptr target_blob(new pcl::PCLPointCloud2), source_blob(new pcl::PCLPointCloud2);;
    toPCLPointCloud2(*input1_keypoints, *source_blob);
    toPCLPointCloud2(*input2_keypoints, *target_blob);

    rej.setSourcePoints(source_blob);
    rej.setSourceNormals(source_blob);
    rej.setTargetPoints(target_blob);
    rej.setTargetNormals(target_blob);

    rej.setInputCorrespondences(temp_correspondences);
    rej.getCorrespondences(correspondences);

    std::cout << "Original Cor: " << temp_correspondences->size() << " R: " << correspondences.size() << std::endl;

    for (int i = 0; i < correspondences.size(); ++i)
    {
        int keypoint = keypoints->at(correspondences[i].index_query);
        int match_keypoint = keypoints2[correspondences[i].index_match];

        pcl::PointXYZRGBNormal p = cloud1->points[keypoint];

        prev2next[keypoint] = match_keypoint;
        pcl::PointXYZRGBNormal transformed_point = cloud2->points[match_keypoint];
        pcl::PointXYZ diff;
        diff.getArray3fMap() = transformed_point.getArray3fMap() - p.getArray3fMap();
        vectors.insert({keypoint, diff});
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed time KP: " << elapsed_secs << std::endl;
}

void MatchFeatures::cloud2_keypoints(std::shared_ptr<std::vector<int>> keypoints,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                                    std::vector<int> &keypoints2) {

    std::vector<int> indices2;
    std::vector<float> squared_distances2;

    std::cout << "Match dist: " << match_radius << std::endl;

    for (int i = 0; i < keypoints->size(); i++) {
        int keypoint = keypoints->at(i);
        pcl::PointXYZRGBNormal p = cloud1->points[keypoint];
        if (tree2.radiusSearch(p, match_radius, indices2, squared_distances2) > 0)
        {
            keypoints2.insert(keypoints2.end(), indices2.begin(), indices2.end());
        }
    }

    std::set<int> s;
    std::size_t size = keypoints2.size();
    for( std::size_t i = 0; i < size; ++i ) s.insert( keypoints2[i] );
    keypoints2.assign( s.begin(), s.end() );
}

Eigen::Matrix4f MatchFeatures::get_best_transformation() {
    return rej.getBestTransformation();
}

void MatchFeatures::local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                                    std::unordered_map<int, pcl::PointXYZ> &vectors,
                                    std::unordered_map<int, int> &prev2next)
{

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    pcl::PointCloud<pcl::SHOT352>::Ptr input1_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input1_keypoints (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    pcl::copyPointCloud (*cloud1, *keypoints, *input1_keypoints);
    descr_est.setInputCloud (input1_keypoints);
    descr_est.setInputNormals (cloud1);
    descr_est.setSearchSurface (cloud1);
    descr_est.compute (*input1_descriptors);


    tree.setInputCloud (cloud1);
    tree2.setInputCloud (cloud2);

    std::vector<int> keypoints2;

    cloud2_keypoints(keypoints, cloud1, cloud2, keypoints2);
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input2_keypoints (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    pcl::PointCloud<pcl::SHOT352>::Ptr input2_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
    
    std::vector<int> indices2;
    std::vector<float> squared_distances2;

    pcl::copyPointCloud (*cloud2, keypoints2, *input2_keypoints);
    descr_est.setInputCloud (input2_keypoints);

    descr_est.setSearchSurface (cloud2);
    descr_est.setInputNormals (cloud2);

    if (input1_descriptors->size () == 0)
        return;

    descr_est.compute (*input2_descriptors);
    
    pcl::CorrespondencesPtr temp_correspondences(new pcl::Correspondences);
    pcl::Correspondences correspondences;
    
    pcl::KdTreeFLANN<pcl::SHOT352> match_search;
    match_search.setInputCloud (input2_descriptors);
    for (size_t i = 0; i < input1_descriptors->size (); ++i)
    {
        std::vector<int> neigh_indices (1);
        std::vector<float> neigh_sqr_dists (1);
        if (!std::isfinite (input1_descriptors->at (i).descriptor[0])) //skipping NaNs
        {
            continue;
        }
        int found_neighs = match_search.nearestKSearch (input1_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
        
        int idx1 = keypoints->at(i), idx2 = neigh_indices[0];

        pcl::PointXYZRGBNormal p = input1_keypoints->points[i], q = input2_keypoints->points[idx2];
        pcl::PointXYZ diff(p.x - q.x, p.y - q.y, p.z - q.z);
        float ddist = norm<pcl::PointXYZ>(diff);
        if(ddist < match_radius) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
        {
            pcl::Correspondence corr (i, neigh_indices[0], ddist);
            temp_correspondences->push_back(corr);
        } else {
            std::cout << "exclude " << ddist << std::endl;
        }
    }

    std::cout << "Kp: " << input1_keypoints->size() << " " << input2_keypoints->size() << std::endl;
    std::cout << "Temp corresp: " << temp_correspondences->size() << std::endl;

    pcl::PCLPointCloud2::Ptr target_blob(new pcl::PCLPointCloud2), source_blob(new pcl::PCLPointCloud2);;
    toPCLPointCloud2(*input1_keypoints, *source_blob);
    toPCLPointCloud2(*input2_keypoints, *target_blob);

    rej.setSourcePoints(source_blob);
    rej.setSourceNormals(source_blob);
    rej.setTargetPoints(target_blob);
    rej.setTargetNormals(target_blob);

    rej.setInputCorrespondences(temp_correspondences);
    rej.getCorrespondences(correspondences);

    std::cout << "Final corresp: " << correspondences.size() << std::endl;

    Eigen::Matrix4f transformation = rej.getBestTransformation();
    
    std::cout << "trasformation:" << std::endl;
    std::cout << transformation << std::endl;
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    pcl::transformPointCloud(*cloud1, *transformed, transformation);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed time : " << elapsed_secs << std::endl;

    for (int i = 0; i < cloud1->size(); ++i)
    {
        pcl::PointXYZRGBNormal p = cloud1->points[i];

        pcl::PointXYZRGBNormal transformed_point = transformed->points[i];
        pcl::PointXYZ diff;
        diff.getArray3fMap() = transformed_point.getArray3fMap() - p.getArray3fMap();
        vectors.insert({i, diff});
    }
}

void MatchTeaser::cloud2_keypoints(std::shared_ptr<std::vector<int>> keypoints,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                                    std::vector<int> &keypoints2) {

    std::vector<int> indices2;
    std::vector<float> squared_distances2;

    std::cout << "Match dist: " << match_radius << std::endl;

    for (int i = 0; i < keypoints->size(); i++) {
        int keypoint = keypoints->at(i);
        pcl::PointXYZRGBNormal p = cloud1->points[keypoint];
        if (tree2.radiusSearch(p, match_radius, indices2, squared_distances2) > 0)
        {
            keypoints2.insert(keypoints2.end(), indices2.begin(), indices2.end());
        }
    }

    std::set<int> s;
    std::size_t size = keypoints2.size();
    for( std::size_t i = 0; i < size; ++i ) s.insert( keypoints2[i] );
    keypoints2.assign( s.begin(), s.end() );
}

Eigen::Matrix4f MatchTeaser::get_best_transformation() {
    return transformation;
}

void MatchTeaser::local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                                    std::unordered_map<int, pcl::PointXYZ> &vectors,
                                    std::unordered_map<int, int> &prev2next)
{
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr input1_descriptors (new pcl::PointCloud<pcl::FPFHSignature33> ());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input1_keypoints (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    pcl::copyPointCloud (*cloud1, *keypoints, *input1_keypoints);
    descr_est.setInputCloud (input1_keypoints);
    descr_est.setInputNormals (cloud1);
    descr_est.setSearchSurface (cloud1);
    descr_est.compute (*input1_descriptors);


    tree.setInputCloud (cloud1);
    tree2.setInputCloud (cloud2);

    std::vector<int> keypoints2;

    cloud2_keypoints(keypoints, cloud1, cloud2, keypoints2);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input2_keypoints (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr input2_descriptors (new pcl::PointCloud<pcl::FPFHSignature33> ());
    
    std::vector<int> indices2;
    std::vector<float> squared_distances2;

    pcl::copyPointCloud (*cloud2, keypoints2, *input2_keypoints);
    descr_est.setInputCloud (input2_keypoints);

    descr_est.setSearchSurface (cloud2);
    descr_est.setInputNormals (cloud2);

    if (input1_descriptors->size () == 0)
        return;

    descr_est.compute (*input2_descriptors);
    
    pcl::CorrespondencesPtr temp_correspondences(new pcl::Correspondences);
    pcl::Correspondences correspondences;
    
    pcl::KdTreeFLANN<pcl::FPFHSignature33> match_search;
    match_search.setInputCloud (input2_descriptors);


    std::vector<std::pair<int, int>> teaser_correspondences;
    teaser_correspondences.reserve(input1_keypoints->size());

    for (size_t i = 0; i < input1_descriptors->size (); ++i)
    {
        std::vector<int> neigh_indices (1);
        std::vector<float> neigh_sqr_dists (1);
        
        int found_neighs = match_search.nearestKSearch (input1_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
        
        int idx1 = keypoints->at(i), idx2 = neigh_indices[0];

        // pcl::PointXYZRGBNormal p = input1_keypoints->points[i], q = input2_keypoints->points[idx2];
        // pcl::PointXYZ diff(p.x - q.x, p.y - q.y, p.z - q.z);
        // float ddist = norm<pcl::PointXYZ>(diff);
        // if(ddist < match_radius) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
        // {
        teaser_correspondences.push_back(std::make_pair(i, idx2));
        // } else {
        //     std::cout << "exclude " << ddist << std::endl;
        // }
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, teaser_correspondences.size());
    for (size_t i = 0; i < teaser_correspondences.size(); ++i) {
        int idx = teaser_correspondences[i].first;
        src.col(i) << input1_keypoints->points[idx].x, input1_keypoints->points[idx].y, input1_keypoints->points[idx].z;
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic> tgt(3, teaser_correspondences.size());
    for (size_t i = 0; i < teaser_correspondences.size(); ++i) {
        int idx = teaser_correspondences[i].second;
        tgt.col(i) << input2_keypoints->points[idx].x, input2_keypoints->points[idx].y, input2_keypoints->points[idx].z;
    }

    solver.solve(src, tgt);
    auto solution = solver.getSolution();

    std::cout << "Final corresp: " << correspondences.size() << std::endl;

    transformation(0,0) = solution.rotation(0,0);
    transformation(0,1) = solution.rotation(0,1);
    transformation(0,2) = solution.rotation(0,2);
    transformation(1,0) = solution.rotation(1,0);
    transformation(1,1) = solution.rotation(1,1);
    transformation(1,2) = solution.rotation(1,2);
    transformation(2,0) = solution.rotation(2,0);
    transformation(2,1) = solution.rotation(2,1);
    transformation(2,2) = solution.rotation(2,2);

    transformation(0,3) = solution.translation(0);
    transformation(1,3) = solution.translation(1);
    transformation(2,3) = solution.translation(2);
    transformation(3,3) = 1;

    std::cout << "trasformation:" << std::endl;
    std::cout << transformation << std::endl;
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    pcl::transformPointCloud(*cloud1, *transformed, transformation);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed time : " << elapsed_secs << std::endl;

    for (int i = 0; i < cloud1->size(); ++i)
    {
        pcl::PointXYZRGBNormal p = cloud1->points[i];

        pcl::PointXYZRGBNormal transformed_point = transformed->points[i];
        pcl::PointXYZ diff;
        diff.getArray3fMap() = transformed_point.getArray3fMap() - p.getArray3fMap();
        vectors.insert({i, diff});
    }
}


void MatchTeaserPoints::cloud2_keypoints(std::shared_ptr<std::vector<int>> keypoints,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                                    std::vector<int> &keypoints2) {

    std::vector<int> indices2;
    std::vector<float> squared_distances2;

    std::cout << "Match dist: " << match_radius << std::endl;

    for (int i = 0; i < keypoints->size(); i++) {
        int keypoint = keypoints->at(i);
        pcl::PointXYZRGBNormal p = cloud1->points[keypoint];
        if (tree2.radiusSearch(p, match_radius, indices2, squared_distances2) > 0)
        {
            keypoints2.insert(keypoints2.end(), indices2.begin(), indices2.end());
        }
    }

    std::set<int> s;
    std::size_t size = keypoints2.size();
    for( std::size_t i = 0; i < size; ++i ) s.insert( keypoints2[i] );
    keypoints2.assign( s.begin(), s.end() );
}

Eigen::Matrix4f MatchTeaserPoints::get_best_transformation() {
    return transformation;
}

void MatchTeaserPoints::local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                                    std::unordered_map<int, pcl::PointXYZ> &vectors,
                                    std::unordered_map<int, int> &prev2next)
{
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input1_keypoints (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    pcl::copyPointCloud (*cloud1, *keypoints, *input1_keypoints);

    tree.setInputCloud (cloud1);
    tree2.setInputCloud (cloud2);

    std::vector<int> keypoints2;

    cloud2_keypoints(keypoints, cloud1, cloud2, keypoints2);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input2_keypoints (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    
    std::vector<int> indices2;
    std::vector<float> squared_distances2;

    pcl::copyPointCloud (*cloud2, keypoints2, *input2_keypoints);

    if (input1_keypoints->size () == 0)
        return;
    
    pcl::CorrespondencesPtr temp_correspondences(new pcl::Correspondences);
    pcl::Correspondences correspondences;

    std::vector<std::pair<int, int>> teaser_correspondences;
    teaser_correspondences.reserve(input1_keypoints->size());

    std::cout << "Matching" << std::endl;
    for (size_t i = 0; i < input1_keypoints->size (); ++i)
    {
        std::vector<int> neigh_indices (1);
        std::vector<float> neigh_sqr_dists (1);
        
        int found_neighs = tree2.nearestKSearch (input1_keypoints->at (i), 1, neigh_indices, neigh_sqr_dists);
        
        int idx1 = keypoints->at(i), idx2 = neigh_indices[0];

        // pcl::PointXYZRGBNormal p = input1_keypoints->points[i], q = input2_keypoints->points[idx2];
        // pcl::PointXYZ diff(p.x - q.x, p.y - q.y, p.z - q.z);
        // float ddist = norm<pcl::PointXYZ>(diff);
        // if(ddist < match_radius) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
        // {
        teaser_correspondences.push_back(std::make_pair(i, idx2));
        // } else {
        //     std::cout << "exclude " << ddist << std::endl;
        // }
    }

    std::cout << "Solving" << std::endl;
    Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, teaser_correspondences.size());
    for (size_t i = 0; i < teaser_correspondences.size(); ++i) {
        int idx = teaser_correspondences[i].first;
        src.col(i) << input1_keypoints->points[idx].x, input1_keypoints->points[idx].y, input1_keypoints->points[idx].z;
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic> tgt(3, teaser_correspondences.size());
    for (size_t i = 0; i < teaser_correspondences.size(); ++i) {
        int idx = teaser_correspondences[i].second;
        tgt.col(i) << cloud2->points[idx].x, cloud2->points[idx].y, cloud2->points[idx].z;
    }

    solver.solve(src, tgt);
    auto solution = solver.getSolution();

    std::cout << "Final corresp: " << correspondences.size() << std::endl;

    transformation(0,0) = solution.rotation(0,0);
    transformation(0,1) = solution.rotation(0,1);
    transformation(0,2) = solution.rotation(0,2);
    transformation(1,0) = solution.rotation(1,0);
    transformation(1,1) = solution.rotation(1,1);
    transformation(1,2) = solution.rotation(1,2);
    transformation(2,0) = solution.rotation(2,0);
    transformation(2,1) = solution.rotation(2,1);
    transformation(2,2) = solution.rotation(2,2);

    transformation(0,3) = solution.translation(0);
    transformation(1,3) = solution.translation(1);
    transformation(2,3) = solution.translation(2);
    transformation(3,3) = 1;

    std::cout << "trasformation:" << std::endl;
    std::cout << transformation << std::endl;
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    pcl::transformPointCloud(*cloud1, *transformed, transformation);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed time : " << elapsed_secs << std::endl;

    for (int i = 0; i < cloud1->size(); ++i)
    {
        pcl::PointXYZRGBNormal p = cloud1->points[i];

        pcl::PointXYZRGBNormal transformed_point = transformed->points[i];
        pcl::PointXYZ diff;
        diff.getArray3fMap() = transformed_point.getArray3fMap() - p.getArray3fMap();
        vectors.insert({i, diff});
    }
}


void MatchKeyCPD::local_correspondences(std::shared_ptr<std::vector<int>> keypoints,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                    std::unordered_map<int, pcl::PointXYZ> &vectors,
                    std::unordered_map<int, int>& prev2next) {
    this->tree.setInputCloud(cloud1);
    this->tree2.setInputCloud(cloud2);

    std::vector<int> indices;
    std::vector<float> squared_distances;
    std::vector<int> indices2;
    std::vector<float> squared_distances2;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr kp_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed_src(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

    std::vector<int> kp(1);

    for (int i = 0; i < static_cast<int>(keypoints->size()); ++i)
    {
        //std::cout << "K: " << i << std::endl;
        pcl::PointXYZRGBNormal &keypoint = cloud1->points[keypoints->at(i)];
        if (tree.radiusSearch(keypoint, this->patch_radius, indices, squared_distances) > 0)
        {
            indices.push_back(keypoints->at(i));
            pcl::copyPointCloud(*cloud1, indices, *src);
            
            int keypoint_idx = keypoints->at(i);

            src->points.push_back(cloud1->points[keypoint_idx]);

            if (tree2.radiusSearch(keypoint, this->patch_radius, indices2, squared_distances2) > 0)
            {
                pcl::copyPointCloud(*cloud2, indices2, *tgt);

                cpd.align(src, tgt, transformed);
                
                pcl::PointXYZRGBNormal src_point = cloud1->points[keypoint_idx];
                pcl::PointXYZRGBNormal transformed_point = transformed->points[transformed->size() - 1];
               
                pcl::PointXYZ diff;

                diff.getArray3fMap() = transformed_point.getArray3fMap() - src_point.getArray3fMap();

                float vec_norm = norm<pcl::PointXYZ>(diff);
                vectors.insert({keypoint_idx, diff});
            }
        }
    }
}
