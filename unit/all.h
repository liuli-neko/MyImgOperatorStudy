// STL
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <vector>

// Eigen
#ifdef USE_EIGEN
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#endif

// OpenCV
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>

using IMG_Mat = cv::Mat;
using Point2d = cv::Point2d;
#endif
// LOG
#define LOG(format, ...)                                                       \
  printf("[%s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)
