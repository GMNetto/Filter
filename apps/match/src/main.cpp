#include <iostream>
#include <unordered_map>
#include <fstream>
#include <chrono>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/log/core.hpp>

#include "util.hpp"
#include "match.hpp"
#include "icp.hpp"
#include "kitti.hpp"

// --------------
// -----Help-----
// --------------
void
printUsage (const char* progName)
{
  std::cout << "\n\nUsage: "<<progName<<" [options]\n\n";
}

int main(int argc, char **argv) {
  
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    printUsage (argv[0]);
    return 0;
  }

  std::string config = "config.json", seg_config = "";
  if (pcl::console::find_argument (argc, argv, "-config") >= 0) {
    pcl::console::parse_argument (argc, argv, "-config", config);
  }

  boost::property_tree::ptree pt, pt_seg;
  boost::property_tree::read_json(config, pt);
  InputParams input_params(pt);
  std::cout << "READ CONFIG" << std::endl;

  std::string tracklet_path = "";

  std::string dir = "/home/gustavo/DT3D/models/Block0.4/", result_file = "./results/";

  int start = 0, end = 4;

  if (pcl::console::find_argument (argc, argv, "-result-file") >= 0)
  {
    pcl::console::parse_argument (argc, argv, "-result-file", result_file);
  }
      
  std::cout << "Match example " << dir << " " << start << " " << end << std::endl;
  readSequence(argc, argv, dir, tracklet_path, start, end);
  
  Tracklets track;
  track.loadFromFile(tracklet_path);

  std::string f1 = dir + std::to_string(start) + "/" + std::to_string(end) + ".txt";
  std::string f2 = dir + std::to_string(start+1) + "/" + "groundless.txt";

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  bool loaded = load_pair(f1, f2, cloud1, cloud2, input_params);
  
  Eigen::Matrix4f transform;

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  if (input_params.kitti_tech == "c2ICP") {
    //CustomICP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
    pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;

    pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointXYZRGBNormal,
        pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr
    cor(new pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>);
    
    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>::Ptr
    rej_sample(new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGBNormal>);
    rej_sample->setMaximumIterations(input_params.ransac_max_iter);
    rej_sample->setInlierThreshold(input_params.match_icp);

    //icp = CustomICP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(input_params); 

    //icp.setCorrespondenceEstimation(cor);
    icp.addCorrespondenceRejector(rej_sample);

    icp.setMaxCorrespondenceDistance(input_params.match_icp);
    icp.setRANSACOutlierRejectionThreshold(input_params.match_icp);
    std::cout << "Transformation Eps: " << input_params.icp_transformation_eps << std::endl;
    icp.setTransformationEpsilon(input_params.icp_transformation_eps);
    icp.setMaximumIterations(input_params.icp_max_iter);
    icp.setRANSACIterations(input_params.ransac_max_iter);

    icp.setInputSource(cloud1);
    icp.setInputTarget(cloud2);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    icp.align(*transformed);
    transform = icp.getFinalTransformation();
  } else if (input_params.kitti_tech == "trICP") {
    pcl::recognition::TrimmedICP<pcl::PointXYZRGBNormal, float> trimmed_icp;
    trimmed_icp.init(cloud2);
    trimmed_icp.setNewToOldEnergyRatio (0.99);
    transform.setIdentity();
    int npo = cloud1->points.size()/5;
    trimmed_icp.align(*cloud1, npo, transform);
  } else if (input_params.kitti_tech == "pmICP") {
    PatchMatchICP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> patchmatch_icp;
    patchmatch_icp = PatchMatchICP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(input_params); 
    pcl::registration::PatchMatchCorrespondenceEstimation<pcl::PointXYZRGBNormal,
    pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr pm_cor(new pcl::registration::PatchMatchCorrespondenceEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(input_params));
    patchmatch_icp.setCorrespondenceEstimation(pm_cor);
    patchmatch_icp.setMaxCorrespondenceDistance(input_params.match_icp);
    patchmatch_icp.setRANSACOutlierRejectionThreshold(input_params.match_icp);
    patchmatch_icp.setTransformationEpsilon(input_params.icp_transformation_eps);
    patchmatch_icp.setMaximumIterations(input_params.icp_max_iter);
    patchmatch_icp.setRANSACIterations(input_params.ransac_max_iter);

    patchmatch_icp.setInputSource(cloud1);
    patchmatch_icp.setInputTarget(cloud2);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    patchmatch_icp.align(*transformed);
    transform = patchmatch_icp.getFinalTransformation();
  } else if (input_params.kitti_tech == "SHOT_RANSAC") {
    MatchFeatures m(input_params);
    Keypoint keypoints(input_params);

    std::shared_ptr<std::vector<int>> key_points = keypoints.get_keypoints(cloud1);
    std::unordered_map<int, int> matches;
    std::unordered_map<int, pcl::PointXYZ> vectors;

    m.local_correspondences(key_points, cloud1, cloud2, vectors, matches);

    transform = m.get_best_transformation();
    //pcl::transformPointCloud(*cloud1, *transformed, transform);
  } else if (input_params.kitti_tech == "TEASER_FEATURE") {
    MatchTeaser m(input_params);
    Keypoint keypoints(input_params);

    std::shared_ptr<std::vector<int>> key_points = keypoints.get_keypoints(cloud1);
    std::unordered_map<int, int> matches;
    std::unordered_map<int, pcl::PointXYZ> vectors;

    m.local_correspondences(key_points, cloud1, cloud2, vectors, matches);

    transform = m.get_best_transformation();
    //pcl::transformPointCloud(*cloud1, *transformed, transform);
  } else if (input_params.kitti_tech == "TEASER_FEATURE_POINT") {
    MatchTeaserPoints m(input_params);
    Keypoint keypoints(input_params);

    std::shared_ptr<std::vector<int>> key_points = keypoints.get_keypoints(cloud1);
    std::unordered_map<int, int> matches;
    std::unordered_map<int, pcl::PointXYZ> vectors;

    m.local_correspondences(key_points, cloud1, cloud2, vectors, matches);

    transform = m.get_best_transformation();
    //pcl::transformPointCloud(*cloud1, *transformed, transform);
  }
  
  std::chrono::steady_clock::time_point finish = std::chrono::steady_clock::now();
  double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(finish - begin).count();
  std::cout << "Elapsed time : " << elapsed_secs << std::endl;

  std::ofstream file_time(result_file + std::to_string(start) + "/time_" + std::to_string(end) + "_" + input_params.kitti_tech + ".txt", std::ofstream::app);
  if (file_time.is_open())
  {
    file_time << elapsed_secs << std::endl;
  }

  file_time.close();

  Tracklets::tPose *pose = new Tracklets::tPose();
  track.getPose(end, start, pose);
  Tracklets::tTracklet *t_track = track.getTracklet(end);
  Eigen::Affine3f retrans = pcl::getTransformation(pose->tx, pose->ty,
  pose->tz + t_track->h / 2.0f, (2*M_PI*pose->rx) - M_PI, (2*M_PI*pose->ry) - M_PI, (2*M_PI*pose->rz) - M_PI);

  std::cout << "Cloud1 transform " << retrans.matrix() << std::endl;
  std::ofstream file1(result_file + std::to_string(start) + "/transform_" + std::to_string(end) + ".txt");
  if (file1.is_open())
  {
    file1 << retrans.matrix() << std::endl;
  }

  std::cout << "Affine transform " << transform.matrix() << std::endl;

  std::ofstream file2(result_file + std::to_string(start) + "/transform_" + std::to_string(end) + "_" + input_params.kitti_tech + ".txt");
  if (file2.is_open())
  {
    file2 << transform << std::endl;
  }

  return 0;  
}
