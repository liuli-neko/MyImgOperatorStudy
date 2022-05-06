cmake_minimum_required(VERSION 3.0.0)

# find opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")


include_directories("/")

add_library(unit unit.cc)
add_library(camera camera.cc)

target_link_libraries(unit ${OpenCV_LIBS})
target_link_libraries(camera ${OpenCV_LIBS})