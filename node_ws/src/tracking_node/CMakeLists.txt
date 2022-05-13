cmake_minimum_required(VERSION 3.0.2)
project(tracking_node)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set (PEDESTRAIN_LIB_DIR src/external/pedestrian_tracker_demo/cpp)
file(GLOB PEDESTRIAN_LIB_SOURCE ${PEDESTRAIN_LIB_DIR}/src/*.cpp)

set(OpenCV_DIR /home/swoopdaddywhoop/intel/openvino_2022/extras/opencv)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  sensor_msgs
  std_msgs
  image_transport
)

# find_package(OpenCV 4.5.5 REQUIRED
#    COMPONENTS core highgui videoio imgproc imgcodecs gapi #Should be optional, tells CMake to use config mode
#    PATHS "/home/swoopdaddywhoop/intel/openvino_2022/extras/opencv"  # Tells CMake to look here
#    NO_DEFAULT_PATH # and don't look anywhere else
# )
find_package(OpenCV 4.5.5 REQUIRED COMPONENTS core highgui videoio imgproc imgcodecs gapi)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_subdirectory(src/external/thirdparty/gflags)
add_subdirectory(src/external/common/cpp)
add_subdirectory(src/external/common/cpp_gapi)
 

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES tracking_node
#  CATKIN_DEPENDS cv_bridge roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PEDESTRAIN_LIB_DIR}/include
)

## Declare a C++ library
add_library(person_tracker_lib STATIC
  src/${PROJECT_NAME}/person_tracker.cpp
  ${PEDESTRIAN_LIB_SOURCE}
)
# target_include_directories(person_tracker_lib PUBLIC 
# "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(person_tracker_lib PUBLIC
  ${OpenCV_LIBRARIES}  
  openvino::runtime
  monitors 
  models 
  pipelines
  utils 
  gflags
)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure


## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(image_converter_node 
  nodes/image_converter_node.cpp
  src/${PROJECT_NAME}/image_converter.cpp
  )

target_link_libraries(image_converter_node PRIVATE
  person_tracker_lib
  ${OpenCV_LIBRARIES}  
  ${catkin_LIBRARIES} 
  )

add_dependencies(image_converter_node 
  ${${PROJECT_NAME}_EXPORTED_TARGETS} 
  ${catkin_EXPORTED_TARGETS}
)


