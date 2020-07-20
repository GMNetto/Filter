#pragma once

#include <time.h>
#include <cmath>

#include <pcl/common/common.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/registration/default_convergence_criteria.h>
#include <pcl/pcl_macros.h>
#include <pcl/registration/eigen.h> 

#include <cpd/nonrigid.hpp>
#include <cpd/gauss_transform_fgt.hpp>

#include <boost/log/trivial.hpp>

#include "util.hpp"
#include "keypoints.hpp"

namespace pcl::registration
{

template <typename PointSource, typename PointTarget, typename Scalar = float>
class PatchMatchCorrespondenceEstimation : public CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>
{
  public:
    typedef boost::shared_ptr<PatchMatchCorrespondenceEstimation<PointSource, PointTarget, Scalar>> Ptr;
    typedef boost::shared_ptr<const PatchMatchCorrespondenceEstimation<PointSource, PointTarget, Scalar>> ConstPtr;

    using CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>::point_representation_;
    using CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>::input_transformed_;
    using CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>::tree_;
    using CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>::tree_reciprocal_;
    using CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>::target_;
    using CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>::corr_name_;
    using CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>::target_indices_;
    using CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>::getClassName;
    using CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>::initCompute;
    using CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>::initComputeReciprocal;
    using CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>::input_;
    using CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>::indices_;
    using CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>::input_fields_;
    using PCLBase<PointSource>::deinitCompute;

    typedef pcl::search::KdTree<PointTarget> KdTree;
    typedef typename pcl::search::KdTree<PointTarget>::Ptr KdTreePtr;

    typedef pcl::PointCloud<PointSource> PointCloudSource;
    typedef typename PointCloudSource::Ptr PointCloudSourcePtr;
    typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;

    typedef pcl::PointCloud<PointTarget> PointCloudTarget;
    typedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
    typedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;

    typedef typename KdTree::PointRepresentationConstPtr PointRepresentationConstPtr;

    pcl::Correspondences min_correspondences;

    float radius, sigma_s=2, sigma_r=0.055, sigma_d=1;
    float min_energy = std::numeric_limits<float>::max();

    PatchMatchCorrespondenceEstimation(InputParams &input_params):CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>() {
        radius = input_params.radius_pf;
        sigma_s = input_params.sigma_s;
        sigma_r = input_params.sigma_r;
        sigma_d = input_params.sigma_d;
        srand (time(NULL));
    } 

    float get_point_correspondence(int index, double max_dist, pcl::Correspondences &t_corresp) {
        std::vector<int> nn_indices;
        std::vector<float> nn_dists;

        //std::cout << "Determine corresp " << index << std::endl;

        const PointSource &p = input_->points[index]; 
        if ( tree_->radiusSearch(p, max_dist, nn_indices, nn_dists) < 1) {
            return std::numeric_limits<float>::max();
        }  

        //std::cout << "# neighbors " << nn_indices.size() << std::endl;
        int random_val = rand() % nn_indices.size();
        int random_index = nn_indices[random_val];

        // It has all points, it shouldn't
        const PointTarget &t = target_->points[random_index]; 
        pcl::Normal v(t.x - p.x, t.y - p.y, t.z - p.z);

        float energy = 0;

        for (int i = 0; i < indices_->size(); i++) {
            //std::cout << "Temp corresp " << i << std::endl;

            PointSource t_p = input_->points[i];
            t_p.x += v.normal_x;
            t_p.y += v.normal_y;
            t_p.z += v.normal_z;

            tree_->nearestKSearch(t_p, 1, nn_indices, nn_dists);

            t_corresp[i].index_query = i;
            t_corresp[i].index_match = nn_indices[0];
            //t_corresp[i].distance = nn_dists[0];

            //float dist = normal_distance(t_p, target_->points[nn_indices[0]]);
            float dist = sqrt(nn_dists[0]);
            t_corresp[i].distance = dist;
            energy += t_corresp[i].distance;
        }
        return energy;
    }

    double normal_distance(PointSource s, PointTarget t) {
        PointTarget pt;
        pt.x = t.x - s.x;
        pt.y = t.y - s.y;
        pt.z = t.z - s.z;

        float norm = norm_normal<PointSource>(s);
        Eigen::Vector3d N (s.normal_x/norm, s.normal_y/norm, s.normal_z/norm);
        Eigen::Vector3d V (pt.x, pt.y, pt.z);
        //return abs(V.dot(N));
        Eigen::Vector3d C = N.cross (V);

        // Check if we have a better correspondence
        return C.dot (C);
    }

    void copyCorresp (pcl::Correspondences &temp, pcl::Correspondences &correspondences) {
        for (int i = 0; i < correspondences.size(); i++) {
            correspondences[i].index_query = temp[i].index_query;
            correspondences[i].index_match = temp[i].index_match;
            correspondences[i].distance = temp[i].distance;
        }
    }

    void
    determineCorrespondences(pcl::Correspondences &correspondences, double max_distance)
    {
        if (!initCompute())
            return;

        srand (time(NULL));
        std::cout << "Determine corresp " << max_distance << std::endl;
        // For each point choose a random_index, and calculate energy of all
        // others, given that translation.
        std::cout << "prev energy: " << min_energy << std::endl;
        
        // Recalculate min-energy
        // if (min_energy != std::numeric_limits<float>::max()) {
        //     min_energy = 0;
        //     for (int i = 0; i < min_correspondences.size(); i++) {
        //         int src = min_correspondences[i].index_query;
        //         int tgt = min_correspondences[i].index_match;
        //         const PointSource p = input_->points[src]; 
        //         const PointTarget t = target_->points[tgt];
        //         min_energy += sqrt(distance<PointSource>(p, t));
        //     }
        //     std::cout << "curr energy: " << min_energy << std::endl;
        // }
        

        correspondences.resize(indices_->size());
        for (int i = 0; i < indices_->size(); i++) {
            pcl::Correspondences temp (indices_->size());
            float energy = get_point_correspondence(i, max_distance, temp);
            //std::cout << "last energy: " << energy << std::endl;
            if (energy < min_energy) {
                if (min_correspondences.size() == 0)
                    min_correspondences.resize(indices_->size());
                copyCorresp(temp, min_correspondences);
                min_energy = energy;
            }
        }
        copyCorresp(min_correspondences, correspondences);
        std::cout << "min energy " << min_energy << std::endl;
        deinitCompute();
    }

};
} // namespace pcl::registration


template <typename PointSource, typename PointTarget, typename Scalar = float>
class PatchMatchICP : public pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>
{
  public:
    typedef typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource PointCloudSource;
    typedef typename PointCloudSource::Ptr PointCloudSourcePtr;
    typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;

    typedef typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget PointCloudTarget;
    typedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
    typedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;

    typedef typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4 Matrix4;
    using pcl::Registration<PointSource, PointTarget, Scalar>::getClassName;
    using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
    using pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>::target_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::nr_iterations_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::max_iterations_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::previous_transformation_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::final_transformation_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::transformation_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::transformation_epsilon_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::converged_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::min_number_correspondences_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::update_visualizer_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::euclidean_fitness_epsilon_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::correspondences_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::transformation_estimation_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::correspondence_estimation_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::correspondence_rejectors_;

    using pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>::need_target_blob_;
    using pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>::target_has_normals_;
    using pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>::convergence_criteria_;
    using pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>::need_source_blob_;
    using pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>::use_reciprocal_correspondence_;
    using pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>::source_has_normals_;
    using pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>::transformCloud;
    using pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>::determineRequiredBlobData;

    typename pcl::registration::CustomConvergenceCriteria<Scalar>::Ptr convergence_criteria_2;


    std::shared_ptr<Keypoint> keypoints;

    PatchMatchICP (InputParams &input_params) : PatchMatchICP() {
        this->keypoints = std::make_shared<Keypoint>(input_params);
    }

    PatchMatchICP () :  pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>() {
        transformation_estimation_.reset (new pcl::registration::TransformationEstimationSVD<PointSource, PointTarget, Scalar> ());
        correspondence_estimation_.reset (new pcl::registration::CorrespondenceEstimation<PointSource, PointTarget, Scalar>);
        convergence_criteria_2.reset(new pcl::registration::CustomConvergenceCriteria<Scalar> (nr_iterations_, transformation_, *correspondences_));
    }

    protected:
    virtual void
    computeTransformation(PointCloudSource &output, const Matrix4 &guess)
    {
        std::cout << "C1 " << input_->size() << std::endl;
        std::cout << "C2 " << target_->size() << std::endl;
        // Point cloud containing the correspondences of each point in <input, indices>
        PointCloudSourcePtr input_transformed(new PointCloudSource);
        PointCloudSourcePtr input_transformed_temp(new PointCloudSource);

        nr_iterations_ = 0;
        converged_ = false;

        // Initialise final transformation to the guessed one
        final_transformation_ = guess;

        // If the guessed transformation is non identity
        if (guess != Matrix4::Identity())
        {
            input_transformed->resize(input_->size());
            // Apply guessed transformation prior to search for neighbours
            transformCloud(*input_, *input_transformed, guess);
        }
        else
            *input_transformed = *input_;

        transformation_ = Matrix4::Identity();

        // Make blobs if necessary
        determineRequiredBlobData();
        pcl::PCLPointCloud2::Ptr target_blob(new pcl::PCLPointCloud2);
        if (need_target_blob_)
            pcl::toPCLPointCloud2(*target_, *target_blob);

        //std::cout << "DEBUG initial corresp " << nr_iterations_ << std::endl;
        // Pass in the default target for the Correspondence Estimation/Rejection code
        correspondence_estimation_->setInputTarget(target_);
        if (correspondence_estimation_->requiresTargetNormals())
            correspondence_estimation_->setTargetNormals(target_blob);
        // Correspondence Rejectors need a binary blob
        for (size_t i = 0; i < correspondence_rejectors_.size(); ++i)
        {
            pcl::registration::CorrespondenceRejector::Ptr &rej = correspondence_rejectors_[i];
            if (rej->requiresTargetPoints())
                rej->setTargetPoints(target_blob);
            if (rej->requiresTargetNormals() && target_has_normals_)
                rej->setTargetNormals(target_blob);
        }

        convergence_criteria_2->setMaximumIterations(max_iterations_);
        convergence_criteria_2->setRelativeMSE(euclidean_fitness_epsilon_);
        convergence_criteria_2->setTranslationThreshold(transformation_epsilon_);
        convergence_criteria_2->setRotationThreshold(1.0 - transformation_epsilon_);
        convergence_criteria_2->setIter(&nr_iterations_);
        convergence_criteria_2->setTransform(&transformation_);
        convergence_criteria_2->setCorrespondence(correspondences_);

        pcl::KdTreeFLANN<PointTarget> tree_;
        //std::cout << "DEBUG set tgt cloud" << nr_iterations_ << std::endl;
        tree_.setInputCloud(target_);
        //std::cout << "DEBUG compute initial error" << input_transformed->size() << std::endl;
        float prev_diff = computeErrorMetric(input_transformed, tree_);

        // Calculate seeds
        std::shared_ptr<std::vector<int>> key_points = keypoints->get_keypoints(input_);
        PointCloudSourcePtr input_sampled(new PointCloudSource);
        // Repeat until convergence
        
        float old_energy, energy = std::numeric_limits<float>::max ();

        //ICP before PM
        pcl::registration::CorrespondenceEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr
        cor(new pcl::registration::CorrespondenceEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>);
        //cor.setMaxCorrespondenceDistance(corr_dist_threshold_);
        
        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>::Ptr
        rej_sample(new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>);
        rej_sample->setMaximumIterations(5);
        rej_sample->setInlierThreshold(corr_dist_threshold_*2);

        cor->setInputTarget(target_);
        if (cor->requiresTargetNormals())
            cor->setTargetNormals(target_blob);
        do
        {
           if (nr_iterations_ >= 0)
               break;
            pcl::PCLPointCloud2::Ptr input_transformed_blob;
            if (need_source_blob_)
            {
                input_transformed_blob.reset(new pcl::PCLPointCloud2);
                toPCLPointCloud2(*input_transformed, *input_transformed_blob);
            }
            previous_transformation_ = transformation_;
            cor->setInputSource(input_transformed);
            if (cor->requiresSourceNormals())
                cor->setSourceNormals(input_transformed_blob);
            std::cout << "Determine ICP corresp" << std::endl;
            cor->determineCorrespondences(*correspondences_, corr_dist_threshold_*2);
           
            pcl::CorrespondencesPtr temp_correspondences(new pcl::Correspondences(*correspondences_));
            *temp_correspondences = *correspondences_;
            if (rej_sample->requiresSourcePoints())
                rej_sample->setSourcePoints(input_transformed_blob);
            if (rej_sample->requiresSourceNormals() && source_has_normals_)
                rej_sample->setSourceNormals(input_transformed_blob);
            rej_sample->setInputCorrespondences(temp_correspondences);
            std::cout << "Determine Rej corresp " << temp_correspondences->size() << std::endl;
            rej_sample->getCorrespondences(*correspondences_);
            // Modify input for the next iteration
            
            std::cout << "Determine first Trans " << correspondences_->size() << std::endl;
            transformation_estimation_->estimateRigidTransformation(*input_transformed, *target_, *correspondences_, transformation_);
            std::cout << "Transform 1" << std::endl << transformation_.matrix() << std::endl;
            pcl::transformPointCloud(*input_transformed, *input_transformed_temp, transformation_);
            
            std::cout << "Calc avg" << std::endl;
            float avg_x = 0, avg_y = 0;
            for (int i = 0; i < input_transformed->size(); i++) {
                PointSource &p = input_transformed->points[i], &q = input_transformed_temp->points[i];
                avg_x += q.x - p.x;
                avg_y += q.y - p.y;
            }
            avg_x /= input_transformed->size();
            avg_y /= input_transformed->size();
            std::cout << "Avg values " <<  avg_x << " " << avg_y << std::endl;
            correspondences_->resize(input_transformed->size());
            for (int i = 0; i < input_transformed->size(); i++) {
                PointSource &p = input_transformed->points[i], &q = input_transformed_temp->points[i];
                q.x = p.x + avg_x;
                q.y = p.y + avg_y;

                (*correspondences_)[i].index_query = i;
                (*correspondences_)[i].index_match = i;
                (*correspondences_)[i].distance = sqrt(q.x*q.x + q.y*q.y);
            }

            transformation_estimation_->estimateRigidTransformation(*input_transformed, *input_transformed_temp, *correspondences_, transformation_);
            transformCloud(*input_transformed, *input_transformed, transformation_);
            
            

            final_transformation_ = transformation_ * final_transformation_;

            ++nr_iterations_;
        } while (!converged_);
        
        // -----------------------
        
        std::cout << "PM" << std::endl;

        nr_iterations_ = 0;
        do
        {
            std::cout << "DEBUG " << nr_iterations_ << std::endl;
            BOOST_LOG_TRIVIAL(debug) << "Iteraction " << nr_iterations_;
           
            if (nr_iterations_ >= max_iterations_)
               break;

            pcl::copyPointCloud (*input_transformed, *key_points, *input_sampled); 

            pcl::PCLPointCloud2::Ptr input_transformed_blob;
            if (need_source_blob_)
            {
                input_transformed_blob.reset(new pcl::PCLPointCloud2);
                toPCLPointCloud2(*input_sampled, *input_transformed_blob);
            }
            // Save the previously estimated transformation
            previous_transformation_ = transformation_;
            //std::cout << "DEBUG1" << std::endl;
            // Set the source each iteration, to ensure the dirty flag is updated
            
            correspondence_estimation_->setInputSource(input_sampled);
            if (correspondence_estimation_->requiresSourceNormals())
                correspondence_estimation_->setSourceNormals(input_transformed_blob);
            // Estimate correspondences
            float max_random_dist = corr_dist_threshold_ * (1.5/ (0.5 + pow(1.5, nr_iterations_)));
            correspondence_estimation_->determineCorrespondences(*correspondences_, max_random_dist);

            pcl::CorrespondencesPtr temp_correspondences(new pcl::Correspondences(*correspondences_));
            for (size_t i = 0; i < correspondence_rejectors_.size(); ++i)
            {
                pcl::registration::CorrespondenceRejector::Ptr &rej = correspondence_rejectors_[i];
                PCL_DEBUG("Applying a correspondence rejector method: %s.\n", rej->getClassName().c_str());
                if (rej->requiresSourcePoints())
                    rej->setSourcePoints(input_transformed_blob);
                if (rej->requiresSourceNormals() && source_has_normals_)
                    rej->setSourceNormals(input_transformed_blob);
                rej->setInputCorrespondences(temp_correspondences);
                rej->getCorrespondences(*correspondences_);
                // Modify input for the next iteration
                if (i < correspondence_rejectors_.size() - 1)
                    *temp_correspondences = *correspondences_;
            }

            size_t cnt = correspondences_->size();
            // Check whether we have enough correspondences
            if (static_cast<int>(cnt) < min_number_correspondences_)
            {
                PCL_ERROR("[pcl::%s::computeTransformation] Not enough correspondences found. Relax your threshold parameters.\n", getClassName().c_str());
                convergence_criteria_2->setConvergenceState(pcl::registration::DefaultConvergenceCriteria<Scalar>::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES);
                converged_ = false;
                break;
            }
            //std::cout << "DEBUG3" << std::endl;
            // Transform sample indexes to source indexes
            // for (int i = 0; i < correspondences_->size(); i++) {
            //     auto &corresp = correspondences_->at(i);
            //     int src_idx = key_points->at(corresp.index_query);
            //     corresp.index_query = src_idx; 
            // }

            float min_energy = 0;
            for (int i = 0; i < correspondences_->size(); i++) {
                int src = correspondences_->at(i).index_query;
                int tgt = correspondences_->at(i).index_match;
                const PointSource p = input_sampled->points[src]; 
                const PointTarget t = target_->points[tgt];
                min_energy += sqrt(distance<PointSource>(p, t));
            }
            std::cout << "curr corresp energy: " << min_energy << std::endl;

            // Estimate the transform
            transformation_estimation_->estimateRigidTransformation(*input_sampled, *target_, *correspondences_, transformation_);
            //std::cout << "DEBUG4" << std::endl;
            // Transform the data
            transformCloud(*input_transformed, *input_transformed, transformation_);

            // Obtain the final transformation
            final_transformation_ = transformation_ * final_transformation_;

            min_energy = 0;
            for (int i = 0; i < correspondences_->size(); i++) {
                int src = correspondences_->at(i).index_query;
                int tgt = correspondences_->at(i).index_match;
                int src_r = key_points->at(src);
                const PointSource p = input_transformed->points[src_r]; 
                const PointTarget t = target_->points[tgt];
                min_energy += sqrt(distance<PointSource>(p, t));
            }
            std::cout << "updated corresp energy: " << min_energy << std::endl;

            ++nr_iterations_;

            // Update the vizualization of icp convergence
            //if (update_visualizer_ != 0)
            //  update_visualizer_(output, source_indices_good, *target_, target_indices_good );

            //std::cout << "CC: " << convergence_criteria_2->getConvergenceState() << " " << pcl::registration::DefaultConvergenceCriteria<Scalar>::CONVERGENCE_CRITERIA_NOT_CONVERGED << std::endl;

            //bool temp = convergence_criteria_2->hasConverged(); 
            //converged_ = temp;
            //std::cout << "CC1: " << temp << " " << converged_ << " " << convergence_criteria_2->getConvergenceState() << std::endl;
        } while (!converged_);

        // Transform the input cloud using the final transformation
        PCL_DEBUG("Transformation is:\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n",
                  final_transformation_(0, 0), final_transformation_(0, 1), final_transformation_(0, 2), final_transformation_(0, 3),
                  final_transformation_(1, 0), final_transformation_(1, 1), final_transformation_(1, 2), final_transformation_(1, 3),
                  final_transformation_(2, 0), final_transformation_(2, 1), final_transformation_(2, 2), final_transformation_(2, 3),
                  final_transformation_(3, 0), final_transformation_(3, 1), final_transformation_(3, 2), final_transformation_(3, 3));

        // Copy all the values
        output = *input_;
        // Transform the XYZ + normals
        transformCloud(*input_, output, final_transformation_);
        //std::cout << "DEBUG5" << std::endl;
    }

    float computeErrorMetric(PointCloudSourceConstPtr cloud, pcl::KdTreeFLANN<PointTarget> &tree_)
    {
        std::vector<int> nn_index(1);
        std::vector<float> nn_distance(1);

        PointSource acc;

        //const ErrorFunctor & compute_error = *error_functor_;
        float error = 0;

        for (int i = 0; i < static_cast<int>(cloud->points.size()); ++i)
        {
            // Find the distance between cloud.points[i] and its nearest neighbor in the target point cloud
            tree_.nearestKSearch(*cloud, i, 1, nn_index, nn_distance);
            // Compute the error
            error += nn_distance[0];
        }
        return error;
    }

};