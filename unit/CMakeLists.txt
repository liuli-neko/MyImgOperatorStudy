cmake_minimum_required(VERSION 3.10.0)

# find opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp")

include_directories("/")

add_library(unit unit.cc)
add_library(camera camera.cc)
add_library(image_operator image_operator.cc)
add_library(sift sift.cc)
add_library(match match.cc)

target_link_libraries(unit ${OpenCV_LIBS})
target_link_libraries(camera ${OpenCV_LIBS})
target_link_libraries(image_operator ${OpenCV_LIBS})
target_link_libraries(sift ${OpenCV_LIBS})
target_link_libraries(match ${OpenCV_LIBS})
target_link_libraries(image_operator unit)
target_link_libraries(sift unit)