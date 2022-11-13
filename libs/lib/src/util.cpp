#include "util.hpp"

#include <fstream>
#include <omp.h>
#include <unordered_set>

#include "normal.hpp"


#include <pcl-1.8/pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/common/common.h>

// --------------------------------------------
// ---------- Read input params ---------------
// --------------------------------------------

bool read2files(int argc, char **argv, std::string &f1, std::string &f2) {
    if (pcl::console::find_argument (argc, argv, "-f1") >= 0) {
      pcl::console::parse_argument (argc, argv, "-f1", f1);
    } else {
      return false;
    }
    if (pcl::console::find_argument (argc, argv, "-f2") >= 0) {
      pcl::console::parse_argument (argc, argv, "-f2", f2);
    } else {
      return false;
    }
    return true;
}

void readSequence(int argc, char **argv, std::string &dir, std::string &tracklet, int &start, int &end) {
  if (pcl::console::find_argument (argc, argv, "-dir") >= 0) {
    pcl::console::parse_argument (argc, argv, "-dir", dir);
  }
  if (pcl::console::find_argument (argc, argv, "-start") >= 0) {
    pcl::console::parse_argument (argc, argv, "-start", start);
  }
  if (pcl::console::find_argument (argc, argv, "-end") >= 0) {
    pcl::console::parse_argument (argc, argv, "-end", end);
  }
  if (pcl::console::find_argument (argc, argv, "-t") >= 0) {
    pcl::console::parse_argument (argc, argv, "-t", tracklet);
  }
}

int load_txt_file(std::string &f, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud) {
  std::ifstream file(f);

  if (!file.is_open()) return -1;

  std::string line;

  while(getline(file, line))
  {

      if (line.empty())
      {
          std::cout << "Empty line." << std::endl;
      }
      else
      {
          std::istringstream tmp(line);
          float x, y, z;

          tmp >> x >> y >> z;
          pcl::PointXYZRGBNormal p;
          p.x = x;
          p.y = y;
          p.z = z;

          cloud->points.push_back(p);
      }
      
  }
  file.close();
  return 0;
}

int save_txt_file(std::string f, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud) {
  std::ofstream outfile;

  std::cout << "Saving " << f << std::endl;
  outfile.open(f, std::fstream::out | std::fstream::trunc);

  for (int i = 0; i < cloud->points.size(); i++) {
    pcl::PointXYZRGBNormal &p = cloud->points[i];
    outfile << p.x << " " << p.y << " " << p.z << std::endl;
  }

  outfile.close();
  return 0;
}

int save_colored_txt_file(std::string f, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud) {
  std::ofstream outfile;

  std::cout << "Saving " << f << std::endl;
  outfile.open(f, std::fstream::out | std::fstream::trunc);

  for (int i = 0; i < cloud->points.size(); i++) {
    pcl::PointXYZRGBNormal &p = cloud->points[i];
    outfile << p.x << " " << p.y << " " << p.z << " " << p.normal_x << " " << p.normal_y << " " << p.normal_z << std::endl;
  }

  outfile.close();
  return 0;
}


bool load_cloud(std::string &f1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                InputParams &input_params) {
    int code0 = pcl::io::loadOBJFile(f1, *cloud1);
    if (code0 >= 0) {
      estimate_normal<pcl::PointXYZRGBNormal>(cloud1, input_params.normal_radius);
      return true;
    }
    code0 = pcl::io::loadPLYFile(f1, *cloud1);
    if (code0 >= 0) {
      estimate_normal<pcl::PointXYZRGBNormal>(cloud1, input_params.normal_radius);
      return true;
    }
    code0 = pcl::io::loadPCDFile(f1, *cloud1);
    if (code0 >= 0) {
      estimate_normal<pcl::PointXYZRGBNormal>(cloud1, input_params.normal_radius);
      return true;
    }
    code0 = load_txt_file(f1, cloud1);
    if (code0 >= 0) {
      estimate_normal<pcl::PointXYZRGBNormal>(cloud1, input_params.normal_radius);
      return true;
    }
    return 0;
}

bool load_pair(std::string &f1, std::string &f2,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                InputParams &input_params) {
    int code0 = pcl::io::loadOBJFile(f1, *cloud1), code1 = pcl::io::loadOBJFile(f2, *cloud2);
    if (code0 >= 0 && code1 >= 0) {
      estimate_normal<pcl::PointXYZRGBNormal>(cloud1, input_params.normal_radius);
      estimate_normal<pcl::PointXYZRGBNormal>(cloud2, input_params.normal_radius);
      return true;
    }
    code0 = pcl::io::loadPLYFile(f1, *cloud1);
    code1 = pcl::io::loadPLYFile(f2, *cloud2);
    if (code0 >= 0 && code1 >= 0) {
      estimate_normal<pcl::PointXYZRGBNormal>(cloud1, input_params.normal_radius);
      estimate_normal<pcl::PointXYZRGBNormal>(cloud2, input_params.normal_radius);
      return true;
    }
    code0 = pcl::io::loadPCDFile(f1, *cloud1);
    code1 = pcl::io::loadPCDFile(f2, *cloud2);
    if (code0 >= 0 && code1 >= 0) {
      estimate_normal<pcl::PointXYZRGBNormal>(cloud1, input_params.normal_radius);
      estimate_normal<pcl::PointXYZRGBNormal>(cloud2, input_params.normal_radius);
      return true;
    }
    code0 = load_txt_file(f1, cloud1);
    code1 = load_txt_file(f2, cloud2);
    if (code0 < 0 || code1 < 0) return false;
    estimate_normal<pcl::PointXYZRGBNormal>(cloud1, input_params.normal_radius);
    estimate_normal<pcl::PointXYZRGBNormal>(cloud2, input_params.normal_radius);
    return true;
}

pcl::PointXYZRGBNormal randomPointParallel(pcl::PointXYZRGBNormal &p, float radius) {
    pcl::PointXYZRGBNormal first, temp, second, norm_n ,direction, result;

    float norm = norm_normal<pcl::PointXYZRGBNormal>(p);
    norm_n.normal_x = p.normal_x / norm;
    norm_n.normal_y = p.normal_y / norm;
    norm_n.normal_z = p.normal_z / norm;

    //std::cout << "norm normal " << norm << std::endl;

    temp.normal_x = norm_n.normal_x + random_float(1, 5);
    temp.normal_y = norm_n.normal_y;
    temp.normal_z = norm_n.normal_z;
    first.getNormalVector3fMap() = norm_n.getNormalVector3fMap().cross(temp.getNormalVector3fMap());

    float angle = random_float(0, 2*M_PI);

    second = rodrigues(first, norm_n, angle);

    //std::cout << "rodrigues " << second << std::endl;

    norm = norm_normal<pcl::PointXYZRGBNormal>(second);
    direction.normal_x = second.normal_x / norm;
    direction.normal_y = second.normal_y / norm;
    direction.normal_z = second.normal_z / norm;

    float dist = random_float(0, radius);

    result.x = p.x + direction.normal_x * dist;
    result.y = p.y + direction.normal_y * dist;
    result.z = p.z + direction.normal_z * dist;
    return result;
}

float random_float(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

pcl::PointXYZRGBNormal rodrigues(pcl::PointXYZRGBNormal &v, pcl::PointXYZRGBNormal &k, float angle) {
    pcl::PointXYZRGBNormal first, second, third, result;
    sum_multi_scalar(v, cos(angle), 0, first);

    second.getNormalVector3fMap() = sin(angle) * k.getNormalVector3fMap().cross(v.getNormalVector3fMap());

    third.getNormalVector3fMap() = k.getNormalVector3fMap() * (1-cos(angle)) * k.getNormalVector3fMap().dot(v.getNormalVector3fMap());

    result.getNormalVector3fMap() = first.getNormalVector3fMap() + second.getNormalVector3fMap() + third.getNormalVector3fMap();

    return result;
}

void join_flows(pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_flow,
           pcl::PointCloud<pcl::PointXYZINormal>::Ptr flow) {
  for (int i = 0; i < icp_flow->size(); i++) {
    pcl::PointXYZINormal &icp_p = icp_flow->points[i], &f_p = flow->points[i];
    f_p.x = icp_p.x;
    f_p.y = icp_p.y;
    f_p.z = icp_p.z;

    f_p.normal_x += icp_p.normal_x;
    f_p.normal_y += icp_p.normal_y;
    f_p.normal_z += icp_p.normal_z; 
  }
}

bool compareNat(const std::string& a, const std::string& b)
{
  if (a.empty())
      return true;
  if (b.empty())
      return false;
  if (std::isdigit(a[0]) && !std::isdigit(b[0]))
      return true;
  if (!std::isdigit(a[0]) && std::isdigit(b[0]))
      return false;
  if (!std::isdigit(a[0]) && !std::isdigit(b[0]))
  {
      if (std::toupper(a[0]) == std::toupper(b[0]))
          return compareNat(a.substr(1), b.substr(1));
      return (std::toupper(a[0]) < std::toupper(b[0]));
  }

  // Both strings begin with digit --> parse both numbers
  std::istringstream issa(a);
  std::istringstream issb(b);
  int ia, ib;
  issa >> ia;
  issb >> ib;
  if (ia != ib)
      return ia < ib;

  // Numbers are the same --> remove numbers and recurse
  std::string anew, bnew;
  std::getline(issa, anew);
  std::getline(issb, bnew);
  return (compareNat(anew, bnew));
}


double interpolate( double val, double y0, double x0, double y1, double x1 ) {
  return (val-x0)*(y1-y0)/(x1-x0) + y0;
}
double blue_jet( double grayscale ) {
  if ( grayscale < -0.33 ) return 1.0;
  else if ( grayscale < 0.33 ) return interpolate( grayscale, 1.0, -0.33, 0.0, 0.33 );
  else return 0.0;
}
double green_jet( double grayscale ) {
  if ( grayscale < -1.0 ) return 0.0; // unexpected grayscale value
  if  ( grayscale < -0.33 ) return interpolate( grayscale, 0.0, -1.0, 1.0, -0.33 );
  else if ( grayscale < 0.33 ) return 1.0;
  else if ( grayscale <= 1.0 ) return interpolate( grayscale, 1.0, 0.33, 0.0, 1.0 );
  else return 1.0; // unexpected grayscale value
}
double red_jet( double grayscale ) {
  if ( grayscale < -0.33 ) return 0.0;
  else if ( grayscale < 0.33 ) return interpolate( grayscale, 0.0, -0.33, 1.0, 0.33 );
  else return 1.0;
}

float l2_SHOT352(const pcl::SHOT352& first, const pcl::SHOT352& second) {
    float squared_diff_sum = 0;
    for (int i = 0; i < 352; i++) {
        squared_diff_sum += pow(first.descriptor[i] - second.descriptor[i], 2);
    }
    return sqrt(squared_diff_sum);
}

float l2_FPFHSignature33(const pcl::FPFHSignature33& first, const pcl::FPFHSignature33& second) {
    float squared_diff_sum = 0;
    for (int i = 0; i < first.descriptorSize(); i++) {
        squared_diff_sum += pow(first.histogram[i] - second.histogram[i], 2);
    }
    return sqrt(squared_diff_sum);
}

int readKittiVelodyne(std::string &fileName, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud){
    std::ifstream input(fileName, std::ios_base::binary);
    if(!input.good()){
        std::cerr<<"Cannot open file : "<<fileName<<std::endl;
        return -1;
    }

    cloud->clear();
    //cloud->height = 1;

    float trash;
    try {
      for (int i=0; input.good() && !input.eof(); i++) {
        pcl::PointXYZRGBNormal point;
        input.read((char *) &point.x, 3*sizeof(float));
        input.read((char *) &trash, sizeof(float));
        cloud->push_back(point);
      }
      std::cerr<<fileName<<":"<<cloud->size()<<" points"<<std::endl;
      input.close();  
    } catch (...) {
      return -1;
    }
    return 0;
}

void readKittiVelodyneIntensity(std::string &fileName, pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud){
    std::ifstream input(fileName, std::ios_base::binary);
    if(!input.good()){
        std::cerr<<"Cannot open file : "<<fileName<<std::endl;
        return;
    }

    cloud->clear();
    //cloud->height = 1;

    //float trash;

    for (int i=0; input.good() && !input.eof(); i++) {
        pcl::PointXYZINormal point;
        input.read((char *) &point.x, 3*sizeof(float));
        input.read((char *) &point.intensity, sizeof(float));
        cloud->push_back(point);
    }
    std::cerr<<fileName<<":"<<cloud->size()<<" points"<<std::endl;
    input.close();
}

pcl::PointXYZ average_vector(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud,
  std::shared_ptr<std::vector<int>> points) {

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZINormal> ());

  if (points) {
    pcl::copyPointCloud (*cloud, *points, *new_cloud);
  } else {
    new_cloud = cloud;
  }
  
  
  pcl::PointXYZ avg_point(0, 0, 0);
  for (int i = 0; i < new_cloud->size(); i++) {
    avg_point.x += new_cloud->points[i].normal_x;
    avg_point.y += new_cloud->points[i].normal_y;
    avg_point.z += new_cloud->points[i].normal_z;
  }
  avg_point.x /= new_cloud->size();
  avg_point.y /= new_cloud->size();
  avg_point.z /= new_cloud->size();
  return avg_point;
}

void sample_cloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
                std::vector<int> &sampled, float radius) {

  pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> tree;
  tree.setInputCloud (cloud2); 

  std::unordered_set<int> s;

  #pragma omp parallel for
  for (int i = 0; i < cloud1->size(); i++) {
      std::vector<int> indices;
      std::vector<float> squared_distances;
      pcl::PointXYZRGBNormal p = cloud1->points[i];
      if (tree.radiusSearch(p, radius, indices, squared_distances) > 0)
      {
        #pragma omp critical
        {
          std::copy(indices.begin(),indices.end(),std::inserter(s,s.end()));
        }
      }
  }

  sampled.assign( s.begin(), s.end() );
}

