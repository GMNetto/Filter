#include <iostream>

#include "visualize.hpp"
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_picking_event.h>
#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types_conversion.h>
#include <cmath>



void camera_pos (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getKeySym () == "c" && event.keyDown ())
  {
    std::cout << "c was pressed => getting camera info" << std::endl;
    std::vector<pcl::visualization::Camera> cam; 
    viewer->getCameras(cam); 

    cout << "Cam: " << endl 
    << " - pos: (" << cam[0].pos[0] << ", "    << cam[0].pos[1] << ", "    << cam[0].pos[2] << ")" << endl 
    << " - view: ("    << cam[0].view[0] << ", "   << cam[0].view[1] << ", "   << cam[0].view[2] << ")"    << endl 
    << " - focal: ("   << cam[0].focal[0] << ", "  << cam[0].focal[1] << ", "  << cam[0].focal[2] << ")"   << endl;
  }
}

// --------------------------------------------
// ----------Visualize Mesh--------------------
// --------------------------------------------
boost::shared_ptr<pcl::visualization::PCLVisualizer> meshVis (const pcl::PolygonMesh& mesh)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (1, 1, 1);
  viewer->addPolygonMesh(mesh,"meshes",0);
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();

  return (viewer);
}

// --------------------------------------------
// ----------VisColor--------------------------
// --------------------------------------------

void color_pick(const pcl::visualization::PointPickingEvent &event,
                         void* viewer_void)
{
    pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
   
    int idx = event.getPointIndex();
    if (idx < 0)
      return;
    float vecx = vectors_c->points[idx].r;
    float vecy = vectors_c->points[idx].g;
    float vecz = vectors_c->points[idx].b;
    std::cout << "Right mouse button released at index (" << idx << "): " << vecx << " " << vecy << " " << vecz << std::endl;
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr vectors_c = NULL;

void pointTest (const pcl::visualization::PointPickingEvent &event,
                         void* viewer_void)
{
    pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
    float x, y, z;
    event.getPoint(x, y, z);
    std::cout << "Right mouse button released at position (" << x << ", " << y << ", " << z << ")" << std::endl;
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (1, 1, 1);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);

  viewer->registerPointPickingCallback (pointTest, (void*)viewer.get ());

  viewer->initCameraParameters ();
  return (viewer);
}

//------------------------------------------------------
//----------------------PointIteraction-----------------
//------------------------------------------------------

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr PointIteraction::cloud = NULL;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr CloudIteraction::cloud = NULL;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr PointIteraction::color = NULL;
std::shared_ptr<CloudIteraction> PointIteraction::cI = NULL;

void PointIteraction::pointPick (const pcl::visualization::PointPickingEvent &event,
                         void* viewer_void)
{
    pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
    float x, y, z;
    event.getPoint(x, y, z);
    std::cout << "Right mouse button released at position (" << x << ", " << y << ", " << z << ")" << std::endl;

    int idx = event.getPointIndex();
    if (idx == -1)
      return;
    color->points[idx].r = 255;
    color->points[idx].g = 0;
    color->points[idx].b = 0;
    std::cout << "Right mouse button released at index (" << idx << ")" << std::endl;
    PointIteraction::cI->markPointIndex(idx);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgbA(color); 
    viewer->updatePointCloud< pcl::PointXYZRGBNormal >(cloud, rgbA, "id");
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> PointIteraction::interactionCustomizationVis ()
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgbA(color); 
    viewer->addPointCloud< pcl::PointXYZRGBNormal >(cloud, rgbA, "id");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "id");
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (1.0);

    viewer->registerPointPickingCallback (PointIteraction::pointPick, (void*)viewer.get ());

    return (viewer);
}


//-------------------------------------------------------------------
//----------------------------FlowVis--------------------------------
//-------------------------------------------------------------------

void flow_vis_loop(pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
  FlowIteraction fI;
  FlowIteraction::vectors = new_cloud;
  //FlowIteraction::cloud = cloud1;

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = fI.flowVis(new_cloud);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_result (new pcl::PointCloud<pcl::PointXYZRGB>);

  while (!viewer->wasStopped ())
  {
      viewer->spinOnce (100);
      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
      boost::mutex::scoped_lock updateLock(FlowIteraction::updateModelMutex);
      if(FlowIteraction::update)
      {
          FlowIteraction::get_colored_cloud(FlowIteraction::vectors, colored_result);
          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(colored_result);
          viewer->updatePointCloud<pcl::PointXYZRGB> (colored_result, rgb, "sample cloud");
          viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
          FlowIteraction::set_camera(viewer);
          
          viewer->registerPointPickingCallback (FlowIteraction::vectors_pick, (void*)viewer.get());

          //viewer->registerKeyboardCallback (camera_pos, (void*)viewer.get ());

          viewer->registerKeyboardCallback (FlowIteraction::update_cloud, (void*)viewer.get());
          std::cout << "updated cloud2" << std::endl;
          FlowIteraction::update = false;
      }
      updateLock.unlock();
  }
}

pcl::PointCloud<pcl::PointXYZINormal>::Ptr FlowIteraction::vectors = NULL;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr FlowIteraction::cloud = NULL;
bool FlowIteraction::update = false;
boost::mutex  FlowIteraction::updateModelMutex;
void FlowIteraction::vectors_pick (const pcl::visualization::PointPickingEvent &event,
                         void* viewer_void)
{
    pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
    int idx = event.getPointIndex();
    if (idx < 0)
      return;
    float vecx = vectors->points[idx].normal_x;
    float vecy = vectors->points[idx].normal_y;
    float vecz = vectors->points[idx].normal_z;
    std::cout << "Right mouse button released at index (" << idx << "): " << vecx << " " << vecy << " " << vecz << std::endl;
}

void FlowIteraction::get_colored_cloud(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_result) {
  pcl::PointCloud<pcl::PointXYZHSV>::Ptr new_cloud (new pcl::PointCloud<pcl::PointXYZHSV>);
  pcl::copyPointCloud(*cloud1, *new_cloud);

  pcl::copyPointCloud(*cloud1, *colored_result);

  float max_mag = 0;
  for (int i = 0; i < cloud1->size(); i++) {
    float vec_x = cloud1->points[i].normal_x;
    float vec_y = cloud1->points[i].normal_y;
    float mag = sqrt(vec_x*vec_x + vec_y*vec_y);
    if (mag > max_mag)
      max_mag = mag;
    float sigma = atan2(vec_y, vec_x);
    new_cloud->points[i].h = sigma*180/(M_PI/2);
    new_cloud->points[i].s = 1;
    new_cloud->points[i].v = mag;
  }

  std::cout << "Max mag " << max_mag << std::endl;

  for (int i = 0; i < cloud1->size(); i++) {
    
    new_cloud->points[i].v /= max_mag;

    pcl::PointXYZHSVtoXYZRGB(new_cloud->points[i], colored_result->points[i]);
    colored_result->points[i].r += 30;
    colored_result->points[i].g += 30;
    colored_result->points[i].b += 30;
  }
}

void FlowIteraction::set_camera(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer) {
  std::vector<pcl::visualization::Camera> cam; 
  viewer->getCameras(cam);


  cam[0].focal[0] = 9.09014;
  cam[0].focal[1] = 3.00311;
  cam[0].focal[2] = 2.26439;

  pcl::visualization::Camera new_camera;

  cam[0].view[0] = 0.195832;
  cam[0].view[1] = 0.069549;
  cam[0].view[2] = 0.978168;

  cam[0].pos[0] = 1.12547;
  cam[0].pos[1] = 3.66064;
  cam[0].pos[2] = 3.81219;


  cam[0].clip[0] = 0.125975;
  cam[0].clip[1] = 125.975;

  viewer->setCameraParameters(cam[0]);
}

void FlowIteraction::set_camera(pcl::visualization::PCLVisualizer *viewer) {

  std::vector<pcl::visualization::Camera> cam; 
  viewer->getCameras(cam);


  cam[0].focal[0] = 9.09014;
  cam[0].focal[1] = 3.00311;
  cam[0].focal[2] = 2.26439;

  pcl::visualization::Camera new_camera;

  cam[0].view[0] = 0.195832;
  cam[0].view[1] = 0.069549;
  cam[0].view[2] = 0.978168;

  cam[0].pos[0] = 1.12547;
  cam[0].pos[1] = 3.66064;
  cam[0].pos[2] = 3.81219;

  //new_camera.focal[0] = 9.09014;
  //new_camera.focal[1] = 3.00311;
  //new_camera.focal[2] = 2.26439;

  cam[0].clip[0] = 0.125975;
  cam[0].clip[1] = 125.975;
  //pcl_vis.updateCamera();
  //viewer->setCameraPosition( 1.12547,  3.66064, 3.81219, 0.195832, 0.069549, 0.978168, 0, 0, 1);

  viewer->setCameraParameters(cam[0]);
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> FlowIteraction::flowVis(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud1) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_result (new pcl::PointCloud<pcl::PointXYZRGB>);
  FlowIteraction::get_colored_cloud(cloud1, colored_result);


  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (1, 1, 1);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(colored_result);
  viewer->addPointCloud<pcl::PointXYZRGB> (colored_result, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();

  FlowIteraction::set_camera(viewer);
  
  viewer->registerPointPickingCallback (vectors_pick, (void*)viewer.get ());

  viewer->registerKeyboardCallback (update_cloud, (void*)viewer.get ());

  return (viewer);
}

void FlowIteraction::update_cloud (const pcl::visualization::KeyboardEvent &event,
                         void* viewer_void) {
  
  //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer2(static_cast<pcl::visualization::PCLVisualizer *> (viewer_void));
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getKeySym () == "n" && event.keyDown ())
  {
    std::cout << "n was pressed => updating cloud" << std::endl;

    boost::mutex::scoped_lock updateLock(updateModelMutex);
    update = true;
    // do processing on cloud
    updateLock.unlock();


    
    std::cout << "updated cloud" << std::endl;
  }
}

//---------------------------------------------------------------------
//----------------------------RegionVis--------------------------------
//---------------------------------------------------------------------

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr RegionInteraction::cloud = NULL;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr RegionInteraction::color = NULL;
std::shared_ptr<CloudIteraction> RegionInteraction::cI = NULL;
bool RegionInteraction::update = false;
boost::mutex  RegionInteraction::updateModelMutex;

void RegionInteraction::color_cloud(SpatialFilter &s_filter
  , pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr colored_result,
    RegionInteraction &instance) {


  pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs (new pcl::PointCloud<pcl::PointXYZINormal>);
  pcl::copyPointCloud(*colored_result, *vecs);
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr result (new pcl::PointCloud<pcl::PointXYZINormal>);
  pcl::copyPointCloud(*colored_result, *result);

  for (int i=0; i < colored_result->size(); i++) {
    vecs->points[i].normal_x = 0;
    vecs->points[i].normal_y = 0;
    vecs->points[i].normal_z = 0;
    vecs->points[i].intensity = 0;
  }

  for (int index: *(cI->points)) {
    std::cout << "Marked in the cloud " << index << std::endl;
    vecs->points[index].normal_x = 1;
    vecs->points[index].normal_y = 0;
    vecs->points[index].normal_z = 0;
    vecs->points[index].intensity = 1;
  }

  s_filter.pf_3D_vec(RegionInteraction::cloud, vecs, result,  *(cI->points));
  
  normalize_colors<pcl::PointXYZINormal>(result, 0.001);

  std::cout << "Defining result" << std::endl;

  for (int i = 0; i < result->size(); i++) {
    float val =  result->points[i].normal_x;
    if (val > 0.1) {
      colored_result->points[i].r = 0;
      colored_result->points[i].g = 0;
      colored_result->points[i].b = 255;
      instance.markPointIndex(i);
    } else {
      colored_result->points[i].r = 100;
      colored_result->points[i].g = 100;
      colored_result->points[i].b = 100;
    }
  }
  for (int index: *(cI->points)) {
    colored_result->points[index].r = 255;
    colored_result->points[index].g = 0;
    colored_result->points[index].b = 0;
  }
  
}

//returns vector of selected points to compare
std::shared_ptr<std::vector<int>>
regionVis(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud,
  SpatialFilter &s_filter) {

  //User interaction
  RegionInteraction pI;
  RegionInteraction::cloud = new_cloud;

  //Cloud which will be colored based on selected points
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr colored_result (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

  pcl::copyPointCloud(*new_cloud, *colored_result);
  RegionInteraction::color = colored_result;

  //Cloud which the user will interact and visulize
  pcl::copyPointCloud(*new_cloud, *(RegionInteraction::cloud));

  CloudIteraction::cloud =  RegionInteraction::cloud;
  std::shared_ptr<CloudIteraction> cI = std::make_shared<CloudIteraction>();   

  RegionInteraction::cI = cI;

  for (int i=0; i < new_cloud->size(); i++) {
    RegionInteraction::cloud->points[i].r = 0;
    RegionInteraction::cloud->points[i].g = 0;
    RegionInteraction::cloud->points[i].b = 0;

    colored_result->points[i].r = 100;
    colored_result->points[i].g = 100;
    colored_result->points[i].b = 100;
  }

  std::cout << "Finished preparing" << std::endl;
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = pI.regionVis();


  while (!viewer->wasStopped ())
  {
      viewer->spinOnce (100);
      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
      boost::mutex::scoped_lock updateLock(RegionInteraction::updateModelMutex);
      if(RegionInteraction::update)
      {
          //Function to get colored cloud
          RegionInteraction::color_cloud(s_filter, colored_result, pI);
          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(colored_result);
          viewer->updatePointCloud<pcl::PointXYZRGBNormal> (colored_result, rgb, "id");
          std::cout << "updated cloud2" << std::endl;
          RegionInteraction::update = false;
      }
      updateLock.unlock();
  }
  return pI.points;
}


void RegionInteraction::point_pick (const pcl::visualization::PointPickingEvent &event,
                         void* viewer_void)
{
    pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
    float x, y, z;
    event.getPoint(x, y, z);
    std::cout << "Right mouse button released at position (" << x << ", " << y << ", " << z << ")" << std::endl;

    int idx = event.getPointIndex();
    if (idx == -1)
      return;
    color->points[idx].r = 255;
    color->points[idx].g = 0;
    color->points[idx].b = 0;
    std::cout << "Right mouse button released at index (" << idx << ")" << std::endl;
    RegionInteraction::cI->markPointIndex(idx);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgbA(color); 
    viewer->updatePointCloud< pcl::PointXYZRGBNormal >(cloud, rgbA, "id");
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> RegionInteraction::regionVis ()
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgbA(color); 
    viewer->addPointCloud< pcl::PointXYZRGBNormal >(cloud, rgbA, "id");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "id");
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (1.0);

    viewer->registerPointPickingCallback (RegionInteraction::point_pick, (void*)viewer.get ());
    viewer->registerKeyboardCallback (RegionInteraction::update_cloud, (void*)viewer.get ());

    return (viewer);
}

void RegionInteraction::update_cloud (const pcl::visualization::KeyboardEvent &event,
                         void* viewer_void) {
  
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getKeySym () == "n" && event.keyDown ())
  {
    std::cout << "n was pressed => updating cloud" << std::endl;

    boost::mutex::scoped_lock updateLock(updateModelMutex);
    update = true;
    // do processing on cloud
    updateLock.unlock();
    std::cout << "updated cloud" << std::endl;
  }
  if (event.getKeySym () == "r" && event.keyDown ())
  {
    std::cout << "r was pressed => reset points" << std::endl;

    boost::mutex::scoped_lock updateLock(updateModelMutex);
    RegionInteraction::cI->reset_points();
    // do processing on cloud
    updateLock.unlock();
    std::cout << "updated cloud" << std::endl;
  }
}
