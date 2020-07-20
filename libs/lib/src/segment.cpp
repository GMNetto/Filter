#include "segment.hpp"

#include "icp.hpp"

float Segment::cluster_angle = 0;
float Segment::cluster_dist = 0;

bool
Segment::customRegionGrowing (const pcl::PointXYZRGBNormal& point_a, const pcl::PointXYZRGBNormal& point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
  if (squared_distance < cluster_dist)
  {
    if (fabs (point_a_normal.dot (point_b_normal)) < cluster_angle)
      return (true);
  }
  return (false);
}

void flow_segment(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr result,
        pcl::IndicesClustersPtr small,
        pcl::IndicesClustersPtr large) {

    pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
    extract.setInputCloud (cloud);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr temp_result (new pcl::PointCloud<pcl::PointXYZINormal>);

    pcl::copyPointCloud(*cloud, *result);
    for (int i = 0; i < result->size(); i++) {
        result->points[i].normal_x = 0;
        result->points[i].normal_y = 0;
        result->points[i].normal_z = 0;
    }

    std::cout << "Test " << small->size () << std::endl; 

    for (int i = 0; i < small->size (); ++i) {
        boost::shared_ptr<std::vector<int> > indicesptr (new std::vector<int> ((*small)[i].indices)); 
        extract.setIndices (indicesptr);
        extract.filter (*temp);
        
        if (temp->size() <= 1)
            continue;
        std::cout << "Cluster ID: " << i << " size " << temp->size() << std::endl;
        icp_clouds(temp, cloud2, 1, temp_result);

        for (int j = 0; j < (*small)[i].indices.size (); ++j) {
            result->points[(*small)[i].indices[j]].normal_x = temp_result->points[j].normal_x;
            result->points[(*small)[i].indices[j]].normal_y = temp_result->points[j].normal_y;
            result->points[(*small)[i].indices[j]].normal_z = temp_result->points[j].normal_z;
        
        }
       
    }
    
}

void Segment::segment(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result) {
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    ground_removal(cloud, result);
    //std::swap(result, new_cloud);
    remove_outliers(result, new_cloud);
    remove_planes(new_cloud, result);
    std::swap(*result, *new_cloud);
    filter_segments(new_cloud, result);
}

void Segment::get_objects(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &results) {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>), result (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    ground_removal(cloud, result);
    //remove_outliers(result, new_cloud);
    //remove_planes(new_cloud, result);
    pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters);
    segment_cloud(result, clusters);
    for (int i = 0; i < clusters->size (); ++i) {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr object (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::copyPointCloud(*result, (*clusters)[i].indices, *object);
        /*for (int j = 0; j < (*clusters)[i].indices.size (); ++j) {
            int cluster_index = (*clusters)[i].indices[j];
            pcl::PointXYZRGBNormal p = result->points[cluster_index];
            object->points.push_back(p);
        }*/
        results.push_back(object);
    }
    
}

void Segment::filter_segments(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result) {
    pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters);
    segment_cloud(cloud, clusters);
    std::cout << "Segments" << clusters->size() << std::endl;
    for (int i = 0; i < clusters->size (); ++i) {
        std::cout << "Clusters seg size " << (*clusters)[i].indices.size () << std::endl;
        for (int j = 0; j < (*clusters)[i].indices.size (); ++j) {
            int cluster_index = (*clusters)[i].indices[j];
            pcl::PointXYZRGBNormal p = cloud->points[cluster_index];
            p.r = 150;
            p.g = 150;
            p.b = 150;
            result->points.push_back(p);
        }
    }
    
}

void Segment::segment_cloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::IndicesClustersPtr clusters) {
    
    cec->setInputCloud (cloud);
    cec->segment (*clusters);
}

void Segment::ground_removal(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result)
{
    pcl::PointIndicesPtr ground (new pcl::PointIndices);

    pcl::ProgressiveMorphologicalFilter<pcl::PointXYZRGBNormal> pmf;
    pmf.setInputCloud (cloud);
    pmf.setMaxWindowSize (ground_size);
    pmf.setSlope (ground_slope);
    pmf.setInitialDistance (0.1f);
    pmf.setMaxDistance (ground_dist);
    pmf.extract (ground->indices);

    // Create the filtering object
    extract->setInputCloud (cloud);
    extract->setIndices (ground);
    extract->filter (*result);
}

void down_sample(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result) {
    pcl::VoxelGrid<pcl::PointXYZRGBNormal> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.06, 0.06, 0.06);
    vg.filter(*result);
}

void Segment::remove_outliers(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result) {
    sor->setInputCloud(cloud);
    sor->filter(*result);
}

void Segment::remove_planes(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result) {
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::copyPointCloud(*cloud, *new_cloud);
    Eigen::Vector3f axis(1, 0, 0);
    seg->setAxis(axis);
    std::cout << "Removing planes" << std::endl;
    for (int i = 0; i < 20;) {
        i++;
        seg->setInputCloud(new_cloud);
        pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr search(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
        seg->segment(*inliers, *coefficients);
        search->setInputCloud (new_cloud); 
        seg->setSamplesMaxDist(1, search);

        if (inliers->indices.size () <= 200) {
            continue; 
        } else {
            i = 0;
        }

        std::cout << "Points in plane found: " << inliers->indices.size () << std::endl;
        
        extract->setInputCloud(new_cloud);
        extract->setIndices(inliers);
        extract->filter(*result);
        pcl::copyPointCloud(*result, *new_cloud);
        std::cout << "New cloud size: " << new_cloud->points.size() << std::endl;
        
    }
    axis = Eigen::Vector3f(0, 1, 0);
    seg->setAxis(axis);
    for (int i = 0; i < 20;) {
        i++;
        seg->setInputCloud(new_cloud);
        pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr search(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
        seg->segment(*inliers, *coefficients);
        search->setInputCloud (new_cloud); 
        seg->setSamplesMaxDist(1, search); 

        if (inliers->indices.size () <= 50) {
            continue; 
        } else {
            i = 0;
        }

        std::cout << "Points in plane found2: " << inliers->indices.size () << std::endl;
        
        extract->setInputCloud(new_cloud);
        extract->setIndices(inliers);
        extract->filter(*result);
        pcl::copyPointCloud(*result, *new_cloud);
        std::cout << "New cloud size: " << new_cloud->points.size() << std::endl;
    }
}

void ground_removal(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result)
{
    pcl::PointIndicesPtr ground (new pcl::PointIndices);

    pcl::ProgressiveMorphologicalFilter<pcl::PointXYZRGBNormal> pmf;
    pmf.setInputCloud (cloud);
    pmf.setMaxWindowSize (1);
    pmf.setSlope (1.0f);
    pmf.setInitialDistance (0.1f);
    pmf.setMaxDistance (0.5f);
    pmf.extract (ground->indices);

    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
    extract.setNegative (true);
    extract.setInputCloud (cloud);
    extract.setIndices (ground);
    extract.filter (*result);

}