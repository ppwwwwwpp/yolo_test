cmake_minimum_required(VERSION 2.8.3)
project(object_detection_yolo)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  sensor_msgs
  std_msgs
  image_transport
  geometry_msgs
  message_filters
)

find_package(Eigen3 REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(InferenceEngine 2.0 REQUIRED)
find_package(ngraph REQUIRED)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES human_pose
#  CATKIN_DEPENDS cv_bridge roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${InferenceEngine_INCLUDE_DIRS}
)

 add_library(monitors STATIC
   include/object_detection_yolo/monitors/presenter.cpp
   include/object_detection_yolo/monitors/cpu_monitor.cpp
   include/object_detection_yolo/monitors/memory_monitor.cpp
   include/object_detection_yolo/monitors/presenter.h
   include/object_detection_yolo/monitors/cpu_monitor.h
   include/object_detection_yolo/monitors/memory_monitor.h
 )

target_link_libraries(monitors PRIVATE opencv_core opencv_imgproc)

add_executable(${PROJECT_NAME}_node 
   src/main.cpp
)
target_link_libraries(${PROJECT_NAME}_node ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES} monitors ${catkin_LIBRARIES} pthread ngraph::ngraph)

