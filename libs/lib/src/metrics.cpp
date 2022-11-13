#include "metrics.hpp"

void crispness_clouds(std::string f1, std::string f2, std::string result_file, InputParams &input_params) {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    bool loaded = load_pair(f1, f2, cloud1, cloud2, input_params);

    std::cout << "Clouds files " << f1 << " " << f2 << std::endl;
    std::cout << "Clouds " << cloud1->size() << " " << cloud2->size() << std::endl;

    std::ofstream outfile, outfile2;

    outfile.open(result_file + ".csv", std::fstream::out | std::fstream::app);
    outfile2.open(result_file + "2.csv", std::fstream::out | std::fstream::app);

    float result = crispness(cloud1, cloud2);
    std::cout << "Result: " << result << std::endl;

    float result2 = crispness3(cloud1, cloud2);
    std::cout << "Result: " << result2 << std::endl;

    if (input_params.use_GT) {
      std::cout << "C1: " << cloud1->size() << " C2: " << cloud2->size() << std::endl;
      std::cout << "Distance c1 -> c2: " << average_of_distance_GT(cloud2, cloud1) << std::endl;
      outfile << std::fixed << std::setprecision(5) << average_of_distance_GT(cloud2, cloud1) << std::endl;
    } else {
      outfile << std::fixed << std::setprecision(4) << result << std::endl;
      outfile2 << std::fixed << std::setprecision(4) << result2 << std::endl;
    }
    
    outfile.close();
    outfile2.close();
}

void crispness_clouds_2(std::string f1, std::string f2, std::string result_file, InputParams &input_params) {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    bool loaded = load_pair(f1, f2, cloud1, cloud2, input_params);

    std::cout << "Clouds files " << f1 << " " << f2 << std::endl;
    std::cout << "Clouds " << cloud1->size() << " " << cloud2->size() << std::endl;

    std::ofstream outfile;

    outfile.open(result_file, std::fstream::out | std::fstream::app);

    float result = crispness3(cloud1, cloud2);
    float result2 = crispness3(cloud2, cloud1);
    float result_avg = (result > result2)? result: result2;
    std::cout << "Result: " << result_avg  << " " << result << " " << result2 << std::endl;
    
    outfile << std::fixed << std::setprecision(4) << result_avg << std::endl;
    
    outfile.close();
}

pcl::PointXYZ crispness(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud,
 pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
 std::shared_ptr<std::vector<int>> points) {

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZINormal> ());

  if (points) {
    pcl::copyPointCloud (*cloud, *points, *new_cloud);
  } else {
    new_cloud = cloud;
  }
  
  pcl::PointXYZ avg_point(0, 0, 0);
  pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> tree;
  tree.setInputCloud (cloud2); 

  std::vector<int> indices(1);
  std::vector<float> squared_distances(1);

  float acc_distances = 0;

  for (int i = 0; i < new_cloud->size(); i++) {
    pcl::PointXYZINormal &p = new_cloud->points[i];
    pcl::PointXYZRGBNormal temp;
    temp.x = p.x;//(p.x + p.normal_x);
    temp.y = p.y;//(p.y + p.normal_y);
    temp.z = p.z;//(p.z + p.normal_z);
    tree.nearestKSearch(temp, 1, indices, squared_distances);
    pcl::PointXYZRGBNormal &n = cloud2->points[indices[0]];
    acc_distances += sqrt(squared_distances[0]);

    avg_point.x += n.x - temp.x;
    avg_point.y += n.y - temp.y;
    avg_point.z += n.z - temp.z;
  }
  avg_point.x /= new_cloud->size();
  avg_point.y /= new_cloud->size();
  avg_point.z /= new_cloud->size();

  std::cout << "Squared distances " << acc_distances << std::endl;

  return avg_point;
}

float crispness3(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
 pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2) {
  
  float acc_distances = 0;

  pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> tree;
  tree.setInputCloud (cloud2);
  std::vector<int> indices(1);
  std::vector<float> squared_distances(1);

  for (int i = 0; i < cloud->size(); i++) {
    pcl::PointXYZRGBNormal &p = cloud->points[i];
    tree.nearestKSearch(p, 1, indices, squared_distances);
    acc_distances += sqrt(squared_distances[0]);

  }

  return acc_distances/cloud->size();
 }

float crispness(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
 pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
 std::shared_ptr<std::vector<int>> points) {

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());

  if (points) {
    pcl::copyPointCloud (*cloud, *points, *new_cloud);
  } else {
    new_cloud = cloud;
  }
  
  pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> tree;
  tree.setInputCloud (cloud2); 

  std::vector<int> indices(1);
  std::vector<float> squared_distances(1);

  float acc_distances = 0;

  for (int i = 0; i < new_cloud->size(); i++) {
    pcl::PointXYZRGBNormal &p = new_cloud->points[i];
    tree.nearestKSearch(p, 1, indices, squared_distances);
    acc_distances += sqrt(squared_distances[0]);

  }

  return acc_distances;
}

float sum_of_distance_GT(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud) {
    
    float dists = 0;
    for (int i = 0; i < cloud2->size(); i++) {
        pcl::PointXYZRGBNormal &p = cloud2->points[i], &n = new_cloud->points[i];
        pcl::PointXYZ diff(p.x - n.x, p.y - n.y, p.z - n.z);
    
        dists += norm<pcl::PointXYZ>(diff);
    }
    return dists;
}

float average_of_distance_GT(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud) {
    
    float dists = 0;
    for (int i = 0; i < cloud2->size(); i++) {
        pcl::PointXYZRGBNormal &p = cloud2->points[i], &n = new_cloud->points[i];
        pcl::PointXYZ diff(p.x - n.x, p.y - n.y, p.z - n.z);
    
        dists += norm<pcl::PointXYZ>(diff);
    }
    return dists/cloud2->size();
}