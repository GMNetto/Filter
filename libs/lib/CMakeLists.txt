cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(point-cloud)
set (CMAKE_CXX_STANDARD 11)

file(GLOB SOURCES "src/*.cpp")

add_library(lib STATIC ${SOURCES})

target_include_directories(lib
  PUBLIC include
  PRIVATE src
)

target_link_libraries (lib PUBLIC ${PCL_LIBRARIES} ${ALGLIB_LIB} ${G2O_LIBS} Boost::log Cpd::Library-C++ cxsparse Fgt::Library-C++ OpenMP::OpenMP_CXX)