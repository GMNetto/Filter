#include "icp.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include <ctime>

void cloud2matrix(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                  cpd::Matrix &matrix) {
    for (int i = 0; i < cloud->points.size(); i++) {
        pcl::PointXYZRGBNormal &p = cloud->points[i];
        matrix(i, 0) = p.x;
        matrix(i, 1) = p.y;
        matrix(i, 2) = p.z;
    }
}

void matrix2cloud(cpd::Matrix &matrix, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud) {
    for (int i = 0; i < matrix.rows(); i++) {
        pcl::PointXYZRGBNormal p;
        p.x = matrix(i, 0);
        p.y = matrix(i, 1);
        p.z = matrix(i, 2);
        cloud->points.push_back(p);
    }
}

void cpd_points(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> points,
                std::shared_ptr<std::vector<int>> points2,
                float match_radius,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    cpd::Matrix matrix(points->size(), 3), matrix2(points2->size(), 3);

    cpd::Nonrigid rigid;
    rigid.beta(2);
    rigid.lambda(2);
    rigid.linked(true);
    rigid.max_iterations(30);
    rigid.tolerance(0.01);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

    pcl::copyPointCloud(*cloud, *points, *src);
    pcl::copyPointCloud(*cloud2, *points2, *tgt);

    cloud2matrix(src, matrix);
    cloud2matrix(tgt, matrix2);

    cpd::NonrigidResult result = rigid.run(matrix2, matrix);

    pcl::copyPointCloud(*cloud, *points, *new_cloud);
    for (int i = 0; i < new_cloud->size(); i++)
    {
        pcl::PointXYZRGBNormal &pt_src = src->points[i];
        new_cloud->points[i].normal_x = result.points(i, 0) - pt_src.x;
        new_cloud->points[i].normal_y = result.points(i, 1) - pt_src.y;
        new_cloud->points[i].normal_z = result.points(i, 2) - pt_src.z;
    }

}

void CPD::align(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud) {
    cpd::Matrix matrix(cloud->size(), 3), matrix2(cloud2->size(), 3);

    cloud2matrix(cloud, matrix);
    cloud2matrix(cloud2, matrix2);

    cpd::NonrigidResult result = nonrigid.run(matrix2, matrix);
    matrix2cloud(result.points, new_cloud);
}

void CPD::getFlow(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
    cpd::Matrix matrix(cloud->size(), 3), matrix2(cloud2->size(), 3);

    cloud2matrix(cloud, matrix);
    cloud2matrix(cloud2, matrix2);
    
    std::clock_t begin = std::clock();

    cpd::NonrigidResult result = nonrigid.run(matrix2, matrix);
    std::clock_t end = std::clock();

    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time: " << elapsed_secs << std::endl;

    pcl::copyPointCloud(*cloud, *new_cloud);
    for (int i = 0; i < new_cloud->size(); i++)
    {
        pcl::PointXYZRGBNormal &pt_src = cloud->points[i];
        new_cloud->points[i].normal_x = result.points(i, 0) - pt_src.x;
        new_cloud->points[i].normal_y = result.points(i, 1) - pt_src.y;
        new_cloud->points[i].normal_z = result.points(i, 2) - pt_src.z;
    }
}

//#define SUPER typename pcl::IterativeClosestPointWithNormals<PointSource, PointTarget, Scalar>
/*
template <typename PointSource, typename PointTarget, typename Scalar>
void CustomICP<PointSource, PointTarget, Scalar>::computeTransformation(
    SUPER1::PointCloudSource &output, const SUPER1::Matrix4 &guess)
{
    // Point cloud containing the correspondences of each point in <input, indices>
    SUPER::PointCloudSourcePtr input_transformed(new SUPER::PointCloudSource);

    nr_iterations_ = 0;
    converged_ = false;

    // Initialise final transformation to the guessed one
    final_transformation_ = guess;

    // If the guessed transformation is non identity
    if (guess != SUPER::Matrix4::Identity())
    {
        input_transformed->resize(input_->size());
        // Apply guessed transformation prior to search for neighbours
        transformCloud(*input_, *input_transformed, guess);
    }
    else
        *input_transformed = *input_;

    transformation_ = SUPER::Matrix4::Identity();

    // Make blobs if necessary
    pcl::IterativeClosestPointWithNormals<PointSource, PointTarget, Scalar>::determineRequiredBlobData();
    pcl::PCLPointCloud2::Ptr target_blob(new pcl::PCLPointCloud2);
    if (need_target_blob_)
        pcl::toPCLPointCloud2(*target_, *target_blob);

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

    convergence_criteria_->setMaximumIterations(max_iterations_);
    convergence_criteria_->setRelativeMSE(euclidean_fitness_epsilon_);
    convergence_criteria_->setTranslationThreshold(transformation_epsilon_);
    convergence_criteria_->setRotationThreshold(1.0 - transformation_epsilon_);

    // Repeat until convergence
    do
    {
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

        converged_ = static_cast<bool>((*convergence_criteria_));
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
}
*/
void icp_points(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::shared_ptr<std::vector<int>> points,
                std::shared_ptr<std::vector<int>> points2,
                float match_radius,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud)
{

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

    pcl::copyPointCloud(*cloud, *points, *src);
    pcl::copyPointCloud(*cloud2, *points2, *tgt);
    pcl::copyPointCloud(*cloud, *points, *new_cloud);

    pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
    icp.setMaximumIterations(15);
    icp.setRANSACIterations(15);

    icp.setMaxCorrespondenceDistance(match_radius);
    icp.setRANSACOutlierRejectionThreshold(match_radius);

    icp.setInputSource(src);
    icp.setInputTarget(tgt);

    icp.setUseReciprocalCorrespondences(true);
    icp.setMaximumIterations(100);
    //icp.setTransformationEpsilon (0.8);
    icp.setMaxCorrespondenceDistance(1);
    //icp.setEuclideanFitnessEpsilon (0.5);
    //icp.setRANSACOutlierRejectionThreshold (0.005);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    icp.align(*transformed);

    //pcl::transformPointCloud (*src, *transformed, icp.getFinalTransformation());

    for (int i = 0; i < new_cloud->size(); i++)
    {
        pcl::PointXYZRGBNormal &pt_src = src->points[i], &pt_trans = transformed->points[i];
        new_cloud->points[i].normal_x = pt_trans.x - pt_src.x;
        new_cloud->points[i].normal_y = pt_trans.y - pt_src.y;
        new_cloud->points[i].normal_z = pt_trans.z - pt_src.z;
    }
}

void icp_clouds(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                float match_radius,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud)
{

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src = cloud;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr tgt = cloud2;

    pcl::copyPointCloud(*cloud, *new_cloud);

    pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
    icp.setMaximumIterations(150);
    icp.setRANSACIterations(150);

    icp.setMaxCorrespondenceDistance(match_radius);
    icp.setRANSACOutlierRejectionThreshold(match_radius);

    icp.setInputSource(src);
    icp.setInputTarget(tgt);

    icp.setUseReciprocalCorrespondences(true);
    icp.setMaximumIterations(100);
    //icp.setTransformationEpsilon (0.8);
    icp.setMaxCorrespondenceDistance(1);
    //icp.setEuclideanFitnessEpsilon (0.5);
    //icp.setRANSACOutlierRejectionThreshold (0.005);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    icp.align(*transformed);

    //pcl::transformPointCloud (*src, *transformed, icp.getFinalTransformation());

    for (int i = 0; i < new_cloud->size(); i++)
    {
        pcl::PointXYZRGBNormal &pt_src = src->points[i], &pt_trans = transformed->points[i];
        new_cloud->points[i].normal_x = pt_trans.x - pt_src.x;
        new_cloud->points[i].normal_y = pt_trans.y - pt_src.y;
        new_cloud->points[i].normal_z = pt_trans.z - pt_src.z;
    }
}
