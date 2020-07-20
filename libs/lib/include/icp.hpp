#pragma once

#include <omp.h>

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

void cpd_points(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> points,
                std::shared_ptr<std::vector<int>> points2,
                float match_radius,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

void icp_points(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> points,
                std::shared_ptr<std::vector<int>> points2,
                float match_radius,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

void icp_clouds(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                float match_radius,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

void cpd_clouds(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

class CPD
{
    private:
    float beta, lambda, tolerance;
    int max_iterations;
    bool linked;

    cpd::Nonrigid nonrigid;

    public:
    CPD(float beta, float lambda, float tolerance, int max_iterations, bool linked):
    beta(beta), lambda(lambda), tolerance(tolerance), max_iterations(max_iterations), linked(linked)
    {
        nonrigid.beta(beta);
        nonrigid.lambda(lambda);
        nonrigid.linked(linked);
        nonrigid.max_iterations(max_iterations);
        nonrigid.tolerance(tolerance);
        nonrigid.gauss_transform(std::move(
        std::unique_ptr<cpd::GaussTransform>(new cpd::GaussTransformFgt())));
    }

    CPD(InputParams &input_params)
    {
        nonrigid.beta(input_params.cpd_beta);
        nonrigid.lambda(input_params.cpd_lambda);
        nonrigid.linked(true);
        nonrigid.max_iterations(input_params.cpd_max_iterations);
        nonrigid.tolerance(input_params.cpd_tolerance);
        nonrigid.gauss_transform(std::move(
        std::unique_ptr<cpd::GaussTransform>(new cpd::GaussTransformFgt())));
    }

    CPD() {
        nonrigid.beta(2);
        nonrigid.lambda(2);
        nonrigid.linked(true);
        nonrigid.max_iterations(30);
        nonrigid.tolerance(5);
        nonrigid.gauss_transform(std::move(
        std::unique_ptr<cpd::GaussTransform>(new cpd::GaussTransformFgt())));
    }

    void getFlow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
               pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
               pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

    void align(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud);
};

namespace pcl::registration
{

template <typename PointSource, typename PointTarget, typename Scalar = float>
class CustomCorrespondenceEstimation : public CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>
{
  public:
    typedef boost::shared_ptr<CustomCorrespondenceEstimation<PointSource, PointTarget, Scalar>> Ptr;
    typedef boost::shared_ptr<const CustomCorrespondenceEstimation<PointSource, PointTarget, Scalar>> ConstPtr;

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

    float radius, sigma_s=2, sigma_r=0.055, sigma_d=1;

    CustomCorrespondenceEstimation(InputParams &input_params):CorrespondenceEstimationNormalShooting<PointSource, PointTarget, Scalar>() {
        radius = input_params.radius_pf;
        sigma_s = input_params.sigma_s;
        sigma_r = input_params.sigma_r;
        sigma_d = input_params.sigma_d;
    } 

    void
    determineCorrespondences(pcl::Correspondences &correspondences, double max_distance)
    {
        int k_ = 15;
        if (!initCompute())
            return;

        double max_dist_sqr = max_distance * max_distance;

        correspondences.resize(indices_->size());

        double min_dist = std::numeric_limits<double>::max();
        int min_index = 0;

        
        unsigned int nr_valid_correspondences = 0;

        // Check if the template types are the same. If true, avoid a copy.
        // Both point types MUST be registered using the POINT_CLOUD_REGISTER_POINT_STRUCT macro!
        if (isSamePointType<PointSource, PointTarget>())
        {
            // Iterate over the input set of source indices
            //for (std::vector<int>::const_iterator idx = indices_->begin(); idx != indices_->end(); ++idx)
            #pragma omp parallel for
            for (int i = 0; i < indices_->size(); i++)
            {
                int idx = indices_->at(i);

                std::vector<int> nn_indices(k_);
                std::vector<float> nn_dists(k_);
                tree_->nearestKSearch(input_->points[idx], k_, nn_indices, nn_dists);


                min_dist = std::numeric_limits<double>::max();

                // Find the best correspondence
                for (size_t j = 0; j < nn_indices.size(); j++)
                {
                    float dist = permeability_radial_vec(target_->points[nn_indices[j]], input_->points[idx]);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        min_index = static_cast<int>(j);
                    }
                }
                if (min_dist > max_distance)
                    continue;

                /*
                if (distance[0] > max_dist_sqr)
                    continue;
                */
                pcl::Correspondence corr;
                corr.index_query = idx;
                corr.index_match = nn_indices[min_index];
                corr.distance = nn_dists[min_index];
                #pragma omp critical
                {
                    correspondences[nr_valid_correspondences++] = corr;
                }
            }
        }
        else
        {
            /*PointTarget pt;

            // Iterate over the input set of source indices
            for (std::vector<int>::const_iterator idx = indices_->begin(); idx != indices_->end(); ++idx)
            {
                // Copy the source data to a target PointTarget format so we can search in the tree
                copyPoint(input_->points[*idx], pt);

                tree_->nearestKSearch(pt, 1, index, distance);
                if (distance[0] > max_dist_sqr)
                    continue;

                corr.index_query = *idx;
                corr.index_match = index[0];
                corr.distance = distance[0];
                correspondences[nr_valid_correspondences++] = corr;
            }*/
        }
        correspondences.resize(nr_valid_correspondences);
        deinitCompute();
    }

    inline float permeability_radial_vec(const PointSource &p, const PointTarget &n)
    {

        float cos_val = dot_product_normal<PointSource, PointTarget>(p, n); // pn.getNormalVector3fMap().dot(nn.getNormalVector3fMap());
        //float cos_val = nn_normal.dot (pn_normal);
        float fdist = acos(std::min(1.0f, cos_val));

        float denominator = M_PI * sigma_r;
        float aux = pow(fdist / denominator, sigma_s);

        pcl::PointXYZ diff(p.x - n.x, p.y - n.y, p.z - n.z);

        float ddist = norm<pcl::PointXYZ>(diff);
        //return sigma_d*ddist + std::min(ddist / (1 + aux), 1.0f);
        return std::min(1 / (1 + aux), 1.0f);
    }
};
} // namespace pcl::registration

namespace pcl::registration {

template <typename Scalar = float>
class CustomConvergenceCriteria : public DefaultConvergenceCriteria<Scalar>
{
    public:
    using Ptr = boost::shared_ptr<CustomConvergenceCriteria<Scalar> >;
    using ConstPtr = boost::shared_ptr<const CustomConvergenceCriteria<Scalar> >;

    using Matrix4 = typename DefaultConvergenceCriteria<Scalar>::Matrix4;
    using DefaultConvergenceCriteria<Scalar>::convergence_state_;
    using DefaultConvergenceCriteria<Scalar>::CONVERGENCE_CRITERIA_NOT_CONVERGED;
    using DefaultConvergenceCriteria<Scalar>::iterations_similar_transforms_;
    using DefaultConvergenceCriteria<Scalar>::iterations_;
    using DefaultConvergenceCriteria<Scalar>::max_iterations_;
    using DefaultConvergenceCriteria<Scalar>::failure_after_max_iter_;
    using DefaultConvergenceCriteria<Scalar>::CONVERGENCE_CRITERIA_ITERATIONS;
    //using DefaultConvergenceCriteria<Scalar>::CONVERGENCE_CRITERIA_FAILURE_AFTER_MAX_ITERATIONS;
    using DefaultConvergenceCriteria<Scalar>::transformation_;
    using DefaultConvergenceCriteria<Scalar>::translation_threshold_;
    using DefaultConvergenceCriteria<Scalar>::rotation_threshold_;
    using DefaultConvergenceCriteria<Scalar>::max_iterations_similar_transforms_;
    using DefaultConvergenceCriteria<Scalar>::CONVERGENCE_CRITERIA_TRANSFORM;
    using DefaultConvergenceCriteria<Scalar>::correspondences_cur_mse_;
    using DefaultConvergenceCriteria<Scalar>::correspondences_prev_mse_;
    using DefaultConvergenceCriteria<Scalar>::mse_threshold_absolute_;
    using DefaultConvergenceCriteria<Scalar>::mse_threshold_relative_;
    using DefaultConvergenceCriteria<Scalar>::CONVERGENCE_CRITERIA_REL_MSE;
    using DefaultConvergenceCriteria<Scalar>::CONVERGENCE_CRITERIA_ABS_MSE;
    using DefaultConvergenceCriteria<Scalar>::correspondences_;
    using DefaultConvergenceCriteria<Scalar>::calculateMSE;

    int *custom_iterations;
    Matrix4 *custom_transform;
    pcl::CorrespondencesPtr custom_correspondences;

    CustomConvergenceCriteria (const int &iterations, const Matrix4 &transform, const pcl::Correspondences &correspondences) :
    DefaultConvergenceCriteria<Scalar>(iterations, transform, correspondences)  {
    }

    void setIter(int *iterations) {
        custom_iterations = iterations;
    }

    void setTransform(Matrix4 *transform) {
        custom_transform = transform;
    }

    void setCorrespondence(pcl::CorrespondencesPtr &correspondences) {
        custom_correspondences = correspondences;
    }

    bool hasConverged ()
    {
        if (convergence_state_ != CONVERGENCE_CRITERIA_NOT_CONVERGED)
        {
            //If it already converged or failed before, reset.
            iterations_similar_transforms_ = 0;
            convergence_state_ = CONVERGENCE_CRITERIA_NOT_CONVERGED;
        }
        
        bool is_similar = false;

        //PCL_DEBUG ("[pcl::DefaultConvergenceCriteria::hasConverged] Iteration %d out of %d.\n", iterations_, max_iterations_);
        
        // 1. Number of iterations has reached the maximum user imposed number of iterations
        if (*custom_iterations >= max_iterations_)
        {
            if (!failure_after_max_iter_)
            {
            convergence_state_ = CONVERGENCE_CRITERIA_ITERATIONS;
            return (true);
            }
            convergence_state_ = CONVERGENCE_CRITERIA_ITERATIONS;
        }

        // 2. The epsilon (difference) between the previous transformation and the current estimated transformation
        double cos_angle = 0.5 * (custom_transform->coeff (0, 0) + custom_transform->coeff (1, 1) + custom_transform->coeff (2, 2) - 1);
        double translation_sqr = custom_transform->coeff (0, 3) * custom_transform->coeff (0, 3) +
                                custom_transform->coeff (1, 3) * custom_transform->coeff (1, 3) +
                                custom_transform->coeff (2, 3) * custom_transform->coeff (2, 3);
        //PCL_DEBUG ("[pcl::DefaultConvergenceCriteria::hasConverged] Current transformation gave %f rotation (cosine) and %f translation.\n", cos_angle, translation_sqr);

        if (cos_angle >= rotation_threshold_ && translation_sqr <= translation_threshold_)
        {
            if (iterations_similar_transforms_ >= max_iterations_similar_transforms_)
            {
            convergence_state_ = CONVERGENCE_CRITERIA_TRANSFORM;
            return (true);
            }
            is_similar = true;
        }

        correspondences_cur_mse_ = calculateMSE (*custom_correspondences);
        //PCL_DEBUG ("[pcl::DefaultConvergenceCriteria::hasConverged] Previous / Current MSE for correspondences distances is: %f / %f.\n", correspondences_prev_mse_, correspondences_cur_mse_);

        // 3. The relative sum of Euclidean squared errors is smaller than a user defined threshold
        // Absolute
        if (std::abs (correspondences_cur_mse_ - correspondences_prev_mse_) < mse_threshold_absolute_)
        {
            if (iterations_similar_transforms_ >= max_iterations_similar_transforms_)
            {
            convergence_state_ = CONVERGENCE_CRITERIA_ABS_MSE;
            return (true);
            }
            is_similar = true;
        }
        
        // Relative
        if (std::abs (correspondences_cur_mse_ - correspondences_prev_mse_) / correspondences_prev_mse_ < mse_threshold_relative_)
        {
            if (iterations_similar_transforms_ >= max_iterations_similar_transforms_)
            {
            convergence_state_ = CONVERGENCE_CRITERIA_REL_MSE;
            return (true);
            }
            is_similar = true;
        }

        if (is_similar)
        {
            // Increment the number of transforms that the thresholds are allowed to be similar
            ++iterations_similar_transforms_;
        }
        else
        {
            // When the transform becomes large, reset.
            iterations_similar_transforms_ = 0;
        }

        correspondences_prev_mse_ = correspondences_cur_mse_;

        return (false);
    }
};

}


template <typename PointSource, typename PointTarget, typename Scalar = float>
class CustomICP : public pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>
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

    CustomICP (InputParams &input_params) : CustomICP() {
    }

    CustomICP () :  pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>() {
        transformation_estimation_.reset (new pcl::registration::TransformationEstimationSVD<PointSource, PointTarget, Scalar> ());
        correspondence_estimation_.reset (new pcl::registration::CorrespondenceEstimation<PointSource, PointTarget, Scalar>);
        convergence_criteria_2.reset(new pcl::registration::CustomConvergenceCriteria<Scalar> (nr_iterations_, transformation_, *correspondences_));
    }

  protected:
    virtual void
    computeTransformation(PointCloudSource &output, const Matrix4 &guess)
    {
        //std::cout << "DEBUG start " << max_iterations_ << std::endl;
        // Point cloud containing the correspondences of each point in <input, indices>
        PointCloudSourcePtr input_transformed(new PointCloudSource);

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

        // Repeat until convergence
        do
        {
            //std::cout << "DEBUG0 " << nr_iterations_ << std::endl;
            //BOOST_LOG_TRIVIAL(debug) << "Iteraction " << nr_iterations_;
            float current_diff = computeErrorMetric(input_transformed, tree_);
            //BOOST_LOG_TRIVIAL(debug) << "Acc error: " << current_diff;
            //std::cout << "Iteraction: " << nr_iterations_;
            //std::cout << " Acc error: " << current_diff << std::endl;
            if (nr_iterations_ > 0 && abs(current_diff - prev_diff) <= 0.001)
               break;
            prev_diff = current_diff;
            // Get blob data if needed
            pcl::PCLPointCloud2::Ptr input_transformed_blob;
            if (need_source_blob_)
            {
                input_transformed_blob.reset(new pcl::PCLPointCloud2);
                toPCLPointCloud2(*input_transformed, *input_transformed_blob);
            }
            // Save the previously estimated transformation
            previous_transformation_ = transformation_;
            //std::cout << "DEBUG1" << std::endl;
            // Set the source each iteration, to ensure the dirty flag is updated
            correspondence_estimation_->setInputSource(input_transformed);
            if (correspondence_estimation_->requiresSourceNormals())
                correspondence_estimation_->setSourceNormals(input_transformed_blob);
            // Estimate correspondences
            if (use_reciprocal_correspondence_)
                correspondence_estimation_->determineReciprocalCorrespondences(*correspondences_, corr_dist_threshold_);
            else
                correspondence_estimation_->determineCorrespondences(*correspondences_, corr_dist_threshold_);
            //std::cout << "DEBUG2" << std::endl;
            //if (correspondence_rejectors_.empty ())
            pcl::CorrespondencesPtr temp_correspondences(new pcl::Correspondences(*correspondences_));
            *temp_correspondences = *correspondences_;
            for (size_t i = 0; i < correspondence_rejectors_.size(); ++i)
            {
                pcl::registration::CorrespondenceRejector::Ptr &rej = correspondence_rejectors_[i];
                //PCL_DEBUG("Applying a correspondence rejector method: %s.\n", rej->getClassName().c_str());
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
            // Estimate the transform
            transformation_estimation_->estimateRigidTransformation(*input_transformed, *target_, *correspondences_, transformation_);
            //std::cout << "DEBUG4" << std::endl;
            // Transform the data
            transformCloud(*input_transformed, *input_transformed, transformation_);

            // Obtain the final transformation
            final_transformation_ = transformation_ * final_transformation_;

            ++nr_iterations_;

            // Update the vizualization of icp convergence
            //if (update_visualizer_ != 0)
            //  update_visualizer_(output, source_indices_good, *target_, target_indices_good );

            //std::cout << "CC: " << convergence_criteria_2->getConvergenceState() << " " << pcl::registration::DefaultConvergenceCriteria<Scalar>::CONVERGENCE_CRITERIA_NOT_CONVERGED << std::endl;

            bool temp = convergence_criteria_2->hasConverged(); 
            converged_ = temp;
            //std::cout << "CC1: " << temp << " " << converged_ << " " << convergence_criteria_2->getConvergenceState() << std::endl;
        } while (!converged_);

        // Transform the input cloud using the final transformation
        // PCL_DEBUG("Transformation is:\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n",
        //           final_transformation_(0, 0), final_transformation_(0, 1), final_transformation_(0, 2), final_transformation_(0, 3),
        //           final_transformation_(1, 0), final_transformation_(1, 1), final_transformation_(1, 2), final_transformation_(1, 3),
        //           final_transformation_(2, 0), final_transformation_(2, 1), final_transformation_(2, 2), final_transformation_(2, 3),
        //           final_transformation_(3, 0), final_transformation_(3, 1), final_transformation_(3, 2), final_transformation_(3, 3));

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


template <typename PointSource, typename PointTarget, typename Scalar = float>
class SimpleICP : public pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>
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

    float radius, sigma_s=2, sigma_r=0.055, sigma_d=1;

    SimpleICP (InputParams &input_params): pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar>() {
        radius = input_params.radius_pf;
        sigma_s = input_params.sigma_s;
        sigma_r = input_params.sigma_r;
        sigma_d = input_params.sigma_d;
    }

  protected:
    virtual void
    computeTransformation(PointCloudSource &output, const Matrix4 &guess)
    {
        // Point cloud containing the correspondences of each point in <input, indices>
        PointCloudSourcePtr input_transformed(new PointCloudSource);

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

        // Pass in the default target for the Correspondence Estimation/Rejection code
        correspondence_estimation_->setInputTarget(target_);
        if (correspondence_estimation_->requiresTargetNormals())
            correspondence_estimation_->setTargetNormals(target_blob);
        // Correspondence Rejectors need a binary blob
        // for (size_t i = 0; i < correspondence_rejectors_.size(); ++i)
        // {
        //     pcl::registration::CorrespondenceRejector::Ptr &rej = correspondence_rejectors_[i];
        //     if (rej->requiresTargetPoints())
        //         rej->setTargetPoints(target_blob);
        //     if (rej->requiresTargetNormals() && target_has_normals_)
        //         rej->setTargetNormals(target_blob);
        // }

        convergence_criteria_->setMaximumIterations(max_iterations_);
        convergence_criteria_->setRelativeMSE(euclidean_fitness_epsilon_);
        convergence_criteria_->setTranslationThreshold(transformation_epsilon_);
        convergence_criteria_->setRotationThreshold(1.0 - transformation_epsilon_);

        pcl::KdTreeFLANN<PointTarget> tree_;
        tree_.setInputCloud(target_);
        float prev_diff = computeErrorMetric(input_transformed, tree_);

        // Repeat until convergence
        do
        {
            //BOOST_LOG_TRIVIAL(debug) << "Iteraction " << nr_iterations_;
            float current_diff = computeErrorMetric(input_transformed, tree_);
            //BOOST_LOG_TRIVIAL(debug) << "Acc error: " << current_diff;
            if (current_diff > prev_diff)
                break;
            prev_diff = current_diff;
            // Get blob data if needed
            pcl::PCLPointCloud2::Ptr input_transformed_blob;
            if (need_source_blob_)
            {
                input_transformed_blob.reset(new pcl::PCLPointCloud2);
                toPCLPointCloud2(*input_transformed, *input_transformed_blob);
            }
            // Save the previously estimated transformation
            previous_transformation_ = transformation_;

            // Set the source each iteration, to ensure the dirty flag is updated
            correspondence_estimation_->setInputSource(input_transformed);
            if (correspondence_estimation_->requiresSourceNormals())
                correspondence_estimation_->setSourceNormals(input_transformed_blob);
            // Estimate correspondences
            if (use_reciprocal_correspondence_)
                correspondence_estimation_->determineReciprocalCorrespondences(*correspondences_, corr_dist_threshold_);
            else
                correspondence_estimation_->determineCorrespondences(*correspondences_, corr_dist_threshold_);

            //if (correspondence_rejectors_.empty ())
            pcl::CorrespondencesPtr temp_correspondences(new pcl::Correspondences(*correspondences_));
            // for (size_t i = 0; i < correspondence_rejectors_.size(); ++i)
            // {
            //     pcl::registration::CorrespondenceRejector::Ptr &rej = correspondence_rejectors_[i];
            //     PCL_DEBUG("Applying a correspondence rejector method: %s.\n", rej->getClassName().c_str());
            //     if (rej->requiresSourcePoints())
            //         rej->setSourcePoints(input_transformed_blob);
            //     if (rej->requiresSourceNormals() && source_has_normals_)
            //         rej->setSourceNormals(input_transformed_blob);
            //     rej->setInputCorrespondences(temp_correspondences);
            //     rej->getCorrespondences(*correspondences_);
            //     // Modify input for the next iteration
            //     if (i < correspondence_rejectors_.size() - 1)
            //         *temp_correspondences = *correspondences_;
            // }

            size_t cnt = correspondences_->size();
            // Check whether we have enough correspondences
            if (static_cast<int>(cnt) < min_number_correspondences_)
            {
                PCL_ERROR("[pcl::%s::computeTransformation] Not enough correspondences found. Relax your threshold parameters.\n", getClassName().c_str());
                convergence_criteria_->setConvergenceState(pcl::registration::DefaultConvergenceCriteria<Scalar>::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES);
                converged_ = false;
                break;
            }

            // Estimate the transform
            transformation_estimation_->estimateRigidTransformation(*input_transformed, *target_, *correspondences_, transformation_);

            // Transform the data
            transformCloud(*input_transformed, *input_transformed, transformation_);

            // Obtain the final transformation
            final_transformation_ = transformation_ * final_transformation_;

            ++nr_iterations_;

            // Update the vizualization of icp convergence
            //if (update_visualizer_ != 0)
            //  update_visualizer_(output, source_indices_good, *target_, target_indices_good );

            convergence_criteria_->setConvergenceState(pcl::registration::DefaultConvergenceCriteria<Scalar>::CONVERGENCE_CRITERIA_ITERATIONS);
            converged_ = true;// static_cast<bool>((*convergence_criteria_));
        } while (!converged_);

        // Transform the input cloud using the final transformation
        // PCL_DEBUG("Transformation is:\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n",
        //           final_transformation_(0, 0), final_transformation_(0, 1), final_transformation_(0, 2), final_transformation_(0, 3),
        //           final_transformation_(1, 0), final_transformation_(1, 1), final_transformation_(1, 2), final_transformation_(1, 3),
        //           final_transformation_(2, 0), final_transformation_(2, 1), final_transformation_(2, 2), final_transformation_(2, 3),
        //           final_transformation_(3, 0), final_transformation_(3, 1), final_transformation_(3, 2), final_transformation_(3, 3));

        // Copy all the values
        output = *input_;
        // Transform the XYZ + normals
        transformCloud(*input_, output, final_transformation_);
    }

    float computeErrorMetric(PointCloudSourceConstPtr cloud, pcl::KdTreeFLANN<PointTarget> &tree_)
    {
        std::vector<int> nn_index(1);
        std::vector<float> nn_distance(1);
        //const ErrorFunctor & compute_error = *error_functor_;
        float error = 0;
        for (int i = 0; i < static_cast<int>(cloud->points.size()); ++i)
        {
            // Find the distance between cloud.points[i] and its nearest neighbor in the target point cloud
            tree_.nearestKSearch(*cloud, i, 1, nn_index, nn_distance);
            const PointSource &cur = cloud->points[i], neighbor = cloud->points[nn_index[0]];

            // Compute the error
            error += nn_distance[0];
        }
        return error;
    }
};
