// STL
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <vector>
#include <algorithm>

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
#endif

#ifndef IMG_OPERATOR_UNIT_ALL_H_
#define IMG_OPERATOR_UNIT_ALL_H_
namespace MY_IMG {
// LOG
#define LOG(format, ...)                                                       \
  printf("[%s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define ASSERT(booleam, format, ...)                                           \
  if (!booleam) {                                                              \
    LOG(format, ##__VA_ARGS__);                                                \
    exit(1);                                                                   \
  }                                                                            \
  void(0)

// eps = 1e-6
const double eps = 1e-6;
struct Point2d {
  int x;
  int y;
  Point2d(int x = 0,int y = 0) : x(x), y(y) {}
  bool operator < (const Point2d &p) const {
    if(x == p.x) {
      return y < p.y;
    }
    return x < p.x;
  }
  bool operator == (const Point2d &p) const {
    return x == p.x && y == p.y;
  }
};
} // namespace MY_IMG

#endif // IMG_OPERATOR_UNIT_ALL_H_