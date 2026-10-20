cmake_minimum_required(VERSION 3.0.2)
project(t800_navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  gazebo_msgs
  nav_msgs
  std_msgs
  std_srvs
  tf
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/gazebo_ground_truth.py
  scripts/waypoint_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
