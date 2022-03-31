cmake_minimum_required(VERSION 3.0.0)

# find opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/")

add_library(img_io img_io.cc)
add_library(unit unit.cc)

target_link_libraries(img_io ${OpenCV_LIBS})
target_link_libraries(unit ${OpenCV_LIBS})