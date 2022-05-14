cmake_minimum_required(VERSION 3.0.0)

# set
set(CMAKE_BUILD_TYPE "Release") # Debug/Release
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)

# add_subdirectory(Eigen3 REQUIRED)
# include_directories(${EIGEN3_INCLUDE_DIRS})

# find opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("../unit")
include_directories("/usr/include")

add_library(myDisplay myDisplay.cpp)
add_library(myMatch myMatch.cpp)
add_library(mySift mySift.cpp)

add_executable(SIFT main.cpp)

target_link_libraries(myDisplay ${OpenCV_LIBS})
target_link_libraries(myMatch ${OpenCV_LIBS} myDisplay)
target_link_libraries(mySift ${OpenCV_LIBS})
target_link_libraries(SIFT ${OpenCV_LIBS})
target_link_libraries(SIFT myDisplay myMatch mySift)

# target_link_libraries(main ${EIGEN3_LIBS})