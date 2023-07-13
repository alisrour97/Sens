cmake_minimum_required(VERSION 3.5)

project(visp_odom)

# Default to C99
if(NOT CMAKE_C_STANDARD)
	set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(VISP REQUIRED)

#################
# Setup targets #
#################
#include_directories(include PUBLIC
#  ${VISP_INCLUDE_DIRS}
#)

add_executable(visp_odom src/visp_odom.cpp)
target_include_directories(visp_odom PUBLIC
  include
  ${VISP_INCLUDE_DIRS}
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  px4_msgs
  Eigen3
  VISP
)

install(TARGETS visp_odom
DESTINATION lib/${PROJECT_NAME})

############
# Install ##
############

# Export information to downstream packages
ament_export_dependencies(ament_cmake rclcpp rosidl_default_runtime eigen3_cmake_module Eigen3 px4_msgs geometry_msgs sensor_msgs)

ament_export_include_directories(include)

# Install header files
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION include/${PROJECT_NAME})


# Install launch files.
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)


ament_package()

