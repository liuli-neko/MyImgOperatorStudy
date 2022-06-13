// STL
#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// OpenCV
#include <opencv2/opencv.hpp>

using IMG_Mat = cv::Mat;

#ifndef IMG_OPERATOR_UNIT_ALL_H_
#define IMG_OPERATOR_UNIT_ALL_H_

namespace MY_IMG {
#ifdef DEBUG
// LOG
#define LOG_DEBUG(fmt, ...)                                                    \
  printf("\033[32m DEBUG\033[0m [%s:%d] \033[1m" fmt "\n\033[0m", __FILE__,    \
         __LINE__, ##__VA_ARGS__)

#define LOG_INFO(fmt, ...)                                                     \
  printf("\033[37m INFO\033[0m [%s:%d] \033[1m" fmt "\n\033[0m", __FILE__,     \
         __LINE__, ##__VA_ARGS__)

#define LOG_WARN(fmt, ...)                                                     \
  printf("\033[33m WARN\033[0m [%s:%d] \033[1m" fmt "\n\033[0m", __FILE__,     \
         __LINE__, ##__VA_ARGS__)

#define LOG_ERROR(fmt, ...)                                                    \
  printf("\033[31m ERROR\033[0m [%s:%d] \033[1m" fmt "\n\033[0m", __FILE__,    \
         __LINE__, ##__VA_ARGS__);                                             \
  exit(1)

#define LOG_FATAL(fmt, ...)                                                    \
  printf("\033[31m FATAL\033[0m [%s:%d] \033[1m" fmt "\n\033[0m", __FILE__,    \
         __LINE__, ##__VA_ARGS__);                                             \
  exit(1)
#define LOG(LEVEL, format, ...) LOG_##LEVEL(format, ##__VA_ARGS__)
#define ASSERT(booleam, format, ...)                                           \
  if (!(booleam)) {                                                            \
    LOG(ERROR, format, ##__VA_ARGS__);                                         \
  }                                                                            \
  void(0)
#define TEST(x) x
#else
#define LOG_DEBUG(fmt, ...) void(0)
#define LOG_INFO(fmt, ...) void(0)
#define LOG_WARN(fmt, ...) void(0)
#define LOG_ERROR(fmt, ...) void(0)
#define LOG_FATAL(fmt, ...) void(0)
#define LOG(LEVEL, format, ...) void(0)
#define ASSERT(booleam, format, ...) void(0)
#endif
#define sqr(x) ((x) * (x))

// eps = 1e-6
const double eps = 1e-6;
struct Point {
  int x;
  int y;
  Point(int x = 0, int y = 0) : x(x), y(y) {}
  bool operator<(const Point &p) const {
    if (x == p.x) {
      return y < p.y;
    }
    return x < p.x;
  }
  bool operator==(const Point &p) const { return x == p.x && y == p.y; }
};
// 用与保存图像金字塔的结构体
struct Octave {
  std::vector<IMG_Mat> layers;
  std::vector<IMG_Mat> dog_layers;
};
// 定义用于存储sift描述子的结构体
struct KeyPoint {
  int x, y;
  int octave;
  int layer;
  float size;
  float angle;
  float response;
  std::vector<float> hist;
  std::vector<float> descriptor;
  KeyPoint(int x = 0, int y = 0, int size = 0, int angle = 0)
      : x(x), y(y), size(size), angle(angle) {}
  bool operator<(const KeyPoint &p) const {
    if (x == p.x) {
      return y < p.y;
    }
    return x < p.x;
  }
  bool operator==(const KeyPoint &p) const { return x == p.x && y == p.y; }
  float operator[](const int &index) const { return descriptor.at(index); }
};
// 定义用于保存原始图片的结构体
struct Image {
  IMG_Mat img;
  int imgId;
  std::vector<std::shared_ptr<KeyPoint>> keypoints;
};
} // namespace MY_IMG

#endif // IMG_OPERATOR_UNIT_ALL_H_