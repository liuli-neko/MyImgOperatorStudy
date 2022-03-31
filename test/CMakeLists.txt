cmake_minimum_required(VERSION 3.0.0)

# set
set(CMAKE_BUILD_TYPE "Debug") # Debug/Release
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)

# find opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUD_DIR})

# find opengl
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${OpenGL_INCLUD_DIRS})
include_directories(${GLUT_INCLUD_DIRS})

add_executable(mat_test mat_test.cc)
add_executable(gl_test gl_test.cc)

target_link_libraries(mat_test ${OpenCV_LIBS})
target_link_libraries(gl_test ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})