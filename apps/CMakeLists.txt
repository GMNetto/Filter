cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(point-cloud)

add_subdirectory(app)
add_subdirectory(temporal_filter)
add_subdirectory(mr_rpm)
add_subdirectory(cpd)
add_subdirectory(match)
add_subdirectory(pf_pos_normal)
add_subdirectory(filter)
add_subdirectory(feature)