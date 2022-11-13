cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(point-cloud)

set (CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake_modules)

find_package(PCL 1.2 REQUIRED)
find_package(Boost COMPONENTS log)
find_package(Fgt REQUIRED)
find_package(Cpd)
find_package(OpenMP)
find_package(ALGLIB REQUIRED)
find_package(Cholmod REQUIRED)
find_package(CSparse REQUIRED)
find_package(G2O REQUIRED)
find_package(teaserpp REQUIRED)

include_directories(include ${PCL_INCLUDE_DIRS} ${ALGLIB_INCLUDE_DIRS} ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} ${Cholmod_INCLUDE_DIR})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_definitions(-DBOOST_ALL_DYN_LINK)
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

LIST(APPEND G2O_LIBS
  cxsparse
  cholmod
  g2o_cli g2o_ext_freeglut_minimal g2o_simulator
  g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d
  g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only
  g2o_types_sba g2o_types_slam3d g2o_csparse_extension
  g2o_opengl_helper g2o_solver_dense g2o_stuff
  g2o_types_sclam2d g2o_parser g2o_solver_pcg
  g2o_types_data g2o_types_sim3
)


add_subdirectory(libs)
add_subdirectory(apps)