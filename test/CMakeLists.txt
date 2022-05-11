cmake_minimum_required(VERSION 3.0.0)

# set
set(CMAKE_BUILD_TYPE "Release") # Debug/Release
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

include_directories("../unit")

add_executable(mat_test mat_test.cc)
add_executable(gl_test gl_test.cc)
add_executable(img_dft img_dft.cc)
add_executable(img_filter img_filter.cc)
add_executable(img_gauss_blur img_gauss_blur.cc)
add_executable(img_histogram_equalization img_histogram_equalization.cc)
add_executable(img_morphology img_morphology.cc)
add_executable(img_opt img_opt.cc)
add_executable(img_sift img_sift.cc)

target_link_libraries(mat_test ${OpenCV_LIBS})
target_link_libraries(gl_test ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
# target_link_libraries(img_dft ${OpenCV_LIBS})
# target_link_libraries(img_filter ${OpenCV_LIBS})
# target_link_libraries(img_gauss_blur ${OpenCV_LIBS})
# target_link_libraries(img_histogram_equalization ${OpenCV_LIBS})
target_link_libraries(img_dft unit)
target_link_libraries(img_dft image_operator)
target_link_libraries(img_filter unit)
target_link_libraries(img_gauss_blur unit)
target_link_libraries(img_histogram_equalization unit)
# target_link_libraries(img_morphology ${OpenCV_LIBS})
target_link_libraries(img_morphology unit)
# target_link_libraries(img_opt unit)
target_link_libraries(img_opt image_operator)
target_link_libraries(img_sift sift)
target_link_libraries(img_sift image_operator)