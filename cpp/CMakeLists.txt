cmake_minimum_required(VERSION 2.8)
#set(OpenCV_DIR /home/koyla/usr/opencv/share/OpenCV/)
set(OpenCV_DIR /opt/libraries/opencv-3.1.0/share/OpenCV)
#set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
project( DisplayImage )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage main.cc )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
