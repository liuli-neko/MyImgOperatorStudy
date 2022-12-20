cmake_minimum_required(VERSION 3.0.0)
project(IMG_OPERATOR VERSION 0.1.0)

# set
# -fsanitize=address
set(CMAKE_BUILD_TYPE "Debug") # Debug/Release
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -W -fsanitize=address -Wall -Wshadow -pthread -fopenmp")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)

add_definitions(-DDEBUG)

# find opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUD_DIR})

# find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIR})

include_directories("unit/")

add_subdirectory(unit)
add_subdirectory(test)
# add_subdirectory(SIFT-opencv-v4.4.0)

add_executable(main main.cc)

target_link_libraries(main unit)
# target_link_libraries(main ${EIGEN3_LIBS})