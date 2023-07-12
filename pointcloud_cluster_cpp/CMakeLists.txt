cmake_minimum_required(VERSION 3.5)
project(pointcloud_cluster)
 
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
 
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(NOT PYBIND11_PYTHON_VERSION)
  set(PYBIND11_PYTHON_VERSION 3.8)
endif()

if(NOT pybind11_DIR)
  set(pybind11_DIR pybind11)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fopenmp) 
endif()

# find_package(ament_cmake REQUIRED)
find_package(pybind11 REQUIRED)
pybind11_add_module(pointcloud_cluster src/PyPointCloud.cpp)

find_package(Eigen3 REQUIRED)
include_directories(
  ${Eigen3_INCLUDE_DIR}
)
target_link_libraries(pointcloud_cluster
PUBLIC
  ${Eigen3_LIBS}
)

find_package(OpenCV REQUIRED)
target_include_directories(pointcloud_cluster
PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(pointcloud_cluster
PUBLIC
  ${OpenCV_LIBS}
)
