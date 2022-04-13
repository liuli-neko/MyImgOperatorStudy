#ifndef IMG_IO_UNIT_H_
#define IMG_IO_UNIT_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

namespace MY_IMG {

#define GRAY_MAX 255
#define LOG(format, ...)                                                       \
  printf("[%s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)

// 拉普拉斯算子
// H1~H5用于图像特征提取
// H6~H9用于图像锐化
const cv::Mat H1 = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
const cv::Mat H2 = (cv::Mat_<double>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
const cv::Mat H3 = (cv::Mat_<double>(3, 3) << 1, -2, 1, -2, 4, -2, 1, -2, 1);
const cv::Mat H4 = (cv::Mat_<double>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
const cv::Mat H5 =
    (cv::Mat_<double>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
const cv::Mat H6 = (cv::Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
const cv::Mat H7 =
    (cv::Mat_<double>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
const cv::Mat H8 = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -5, 1, 0, 1, 0);
const cv::Mat H9 = (cv::Mat_<double>(3, 3) << 1, 1, 1, 1, -9, 1, 1, 1, 1);

// 两点之间的距离
inline double Dist(double x1, double y1, double x2, double y2) {
  return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

// 巴特沃斯算子
struct ButterworthFilter {
  double D0;
  int n;
  double W;
  std::pair<double, double> center;
  ButterworthFilter(std::pair<double, double> center, double D0 = 10, int n = 2,
                    double W = 1.0)
      : center(center), D0(D0), n(n), W(W) {}
  double LowPassFilter(double x, double y) {
    double D = Dist(x, y, center.first, center.second);
    return 1.0 / (1.0 + pow(D / D0, 2 * n));
  }
  double HighPassFilter(double x, double y) {
    double D = Dist(x, y, center.first, center.second);
    return 1.0 / (1.0 + pow(D0 / D, 2 * n));
  }
  double BandPassFilter(double x, double y) {
    double D = Dist(x, y, center.first, center.second);
    return 1.0 / (1.0 + pow(D * W / (D * D - D0 * D0), 2 * n));
  }
};

// 高斯算子
struct GaussianFilter {
  double D0;
  std::pair<double, double> center;
  GaussianFilter(std::pair<double, double> center, double D0 = 1.0)
      : center(center), D0(D0) {}
  double LowPassFilter(double x, double y) {
    double D = Dist(x, y, center.first, center.second);
    return exp(-D * D / (2 * D0 * D0));
  }
  double HighPassFilter(double x, double y) {
    double D = Dist(x, y, center.first, center.second);
    return 1 - exp(-D * D / (2 * D0 * D0));
  }
};
/**
 * @brief 图像灰度化
 * @param src 原图像
 * @param dst 灰度化后的图像
 */
void Rgb2Gray(const cv::Mat &src, cv::Mat &dst);
/**
 * @brief 对图像进行直方图均衡化
 * @param src 原图像
 * @param dst 均衡化后的图像
 */
void HistogramEqualization(const cv::Mat &src, cv::Mat &dst);
/**
 * @brief 创建高斯滤波卷积核
 * @param filter 滤波核
 * @param sigma 标准差
 * @param radim 滤波核半径
 * @note 滤波核半径为-1时将自动计算
 */
void CreateGaussBlurFilter(cv::Mat &filter, const double &sigma,
                           int radim = -1);
void CreateGaussBlurFilter(cv::Mat &filter, const double &sigma, int radim_h,
                           int radim_w);
/**
 * @brief 对图像进行卷积
 * @param src 原图像
 * @param dst 卷积后的图像
 * @param filter 滤波核
 * @param is_resverse 是否反转卷积后的图像
 */
void ImgFilter(const cv::Mat &img, const cv::Mat &filter, cv::Mat &dst,
               const bool &is_resverse = false);

// 转换图像存储格式
cv::Mat ConvertComplexMat2doubleMat(const cv::Mat &img);
template <typename T>
cv::Mat ConvertSingleChannelMat2ComplexMat(const cv::Mat &img);
cv::Mat ConvertDoubleMat2Uint8Mat(const cv::Mat &img,const bool &is_mapping = false);

// 定义傅里叶变换的函数声明
void DFT(const cv::Mat &src, cv::Mat &dst);
void IDFT(const cv::Mat &src, cv::Mat &dst);
void FFT2D(const cv::Mat &src, cv::Mat &dst);
void IFFT2D(const cv::Mat &src, cv::Mat &dst);

// 形态学操作
cv::Mat GrayCorrosion(const cv::Mat &src,const cv::Mat &struct_element);
cv::Mat GrayExpansion(const cv::Mat &src,const cv::Mat &struct_element);
cv::Mat GrayOpening(const cv::Mat &src,const cv::Mat &struct_element);
cv::Mat GrayClosing(const cv::Mat &src,const cv::Mat &struct_element);

} // namespace MY_IMG
template <typename T>
cv::Mat MY_IMG::ConvertSingleChannelMat2ComplexMat(const cv::Mat &img) {
  int height = img.rows;
  int width = img.cols;
  cv::Mat result = cv::Mat(height, width, CV_64FC2);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      result.at<std::complex<double>>(i, j) =
          std::complex<double>(static_cast<double>(img.at<T>(i, j)), 0);
    }
  }
  return result;
}
#endif