#ifndef IMG_OPERATOR_UNIT_UNTI_H_
#define IMG_OPERATOR_UNIT_UNTI_H_
#define USE_OPENCV
#define USE_EIGEN
#include "all.h"

namespace MY_IMG {

#define GRAY_MAX 255

// 拉普拉斯算子
// H1~H5用于图像特征提取
// H6~H9用于图像锐化
const IMG_Mat H1 = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
const IMG_Mat H2 = (cv::Mat_<double>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
const IMG_Mat H3 = (cv::Mat_<double>(3, 3) << 1, -2, 1, -2, 4, -2, 1, -2, 1);
const IMG_Mat H4 = (cv::Mat_<double>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
const IMG_Mat H5 =
    (cv::Mat_<double>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
const IMG_Mat H6 = (cv::Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
const IMG_Mat H7 =
    (cv::Mat_<double>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
const IMG_Mat H8 = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -5, 1, 0, 1, 0);
const IMG_Mat H9 = (cv::Mat_<double>(3, 3) << 1, 1, 1, 1, -9, 1, 1, 1, 1);

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
void Rgb2Gray(const IMG_Mat &src, IMG_Mat &dst,
              const std::vector<double> &w = {0.299, 0.587, 0.114});
/**
 * @brief 对图像进行直方图均衡化
 * @param src 原图像
 * @param dst 均衡化后的图像
 */
void HistogramEqualization(const IMG_Mat &src, IMG_Mat &dst);
/**
 * @brief 创建高斯滤波卷积核
 * @param filter 滤波核
 * @param sigma 标准差
 * @param radim 滤波核半径
 * @note 滤波核半径为-1时将自动计算
 */
void CreateGaussBlurFilter(IMG_Mat &filter, const double &sigma,
                           int radim = -1);
void CreateGaussBlurFilter(IMG_Mat &filter, const double &sigma, int radim_h,
                           int radim_w);
/**
 * @brief 对图像进行卷积
 * @param src 原图像
 * @param dst 卷积后的图像
 * @param filter 滤波核
 * @param is_resverse 是否反转卷积后的图像
 */
void ImgFilter(const IMG_Mat &img, const IMG_Mat &filter, IMG_Mat &dst,
               const bool &is_resverse = false);

// 转换图像存储格式
IMG_Mat ConvertComplexMat2doubleMat(const IMG_Mat &img);
template <typename T>
IMG_Mat ConvertSingleChannelMat2ComplexMat(const IMG_Mat &img);
IMG_Mat ConvertDoubleMat2Uint8Mat(const IMG_Mat &img,
                                  const bool &is_mapping = false);

// 定义傅里叶变换的函数声明
void DFT(const IMG_Mat &src, IMG_Mat &dst);
void IDFT(const IMG_Mat &src, IMG_Mat &dst);
void FFT2D(const IMG_Mat &src, IMG_Mat &dst);
void IFFT2D(const IMG_Mat &src, IMG_Mat &dst);

// 形态学操作
IMG_Mat GrayCorrosion(const IMG_Mat &src, const IMG_Mat &struct_element);
IMG_Mat GrayExpansion(const IMG_Mat &src, const IMG_Mat &struct_element);
IMG_Mat GrayOpening(const IMG_Mat &src, const IMG_Mat &struct_element);
IMG_Mat GrayClosing(const IMG_Mat &src, const IMG_Mat &struct_element);

} // namespace MY_IMG
template <typename T>
IMG_Mat MY_IMG::ConvertSingleChannelMat2ComplexMat(const IMG_Mat &img) {
  int height = img.rows;
  int width = img.cols;
  IMG_Mat result = IMG_Mat(height, width, CV_64FC2);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      result.at<std::complex<double>>(i, j) =
          std::complex<double>(static_cast<double>(img.at<T>(i, j)), 0);
    }
  }
  return result;
}
#endif