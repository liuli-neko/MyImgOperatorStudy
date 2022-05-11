#ifndef IMG_OPERATOR_UNIT_IMAGE_OPERATOR_H_
#define IMG_OPERATOR_UNIT_IMAGE_OPERATOR_H_
#define USE_OPENCV
#include "all.h"
namespace MY_IMG {
/**
 * @brief 生成图像的高斯模糊集
 * @param img 待处理的图像
 * @param img_out 输出的图像集
 * @param sigma 频率域滤波半径集
 */
void ImageGaussianFilter(const IMG_Mat &img, std::vector<IMG_Mat> &img_out,
                         const std::vector<double> &sigma);
/**
 * @brief 图像旋转
 * @param[in] src 原图像
 * @param[out] dst 旋转后的图像
 * @param[in] zoom 缩放比例
 * @param[in] angle 旋转角度(弧度)
 */
void ImageChange(const IMG_Mat &img, IMG_Mat &img_out, const double &zoom = 2,
                 const double &angle = 0);
void DrawPoints(const IMG_Mat &img, const std::vector<SiftPointDescriptor> &keypoints,
                IMG_Mat &img_out);
template <typename InPixeType, typename OutPixeType>
void ConvertTo(const IMG_Mat &img, IMG_Mat &img_out,
               const double min_value = 0.0, const double max_value = 255.0);
} // namespace MY_IMG

template <typename InPixeType, typename OutPixeType>
void MY_IMG::ConvertTo(const IMG_Mat &img, IMG_Mat &img_out,
                       const double min_value, const double max_value) {
  ASSERT(img_out.empty() == false, "ConvertTo: img is empty");
  ASSERT(img.cols <= img_out.cols && img.rows <= img_out.rows,
         "ConvertTo: img size[%d %d] is not equal to img_out[%d %d]", img.cols,
         img.rows, img_out.cols, img_out.rows);
  double min_val = 1e9, max_val = -1e9;
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      min_val =
          std::min(min_val, static_cast<double>(img.at<InPixeType>(i, j)));
      max_val =
          std::max(max_val, static_cast<double>(img.at<InPixeType>(i, j)));
    }
  }
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      double value = static_cast<double>(img.at<InPixeType>(i, j)) *
                         (max_value - min_value) / (max_val - min_val) +
                     min_value;
      if (value < min_value) {
        value = min_value;
      }
      if (value > max_value) {
        value = max_value;
      }
      img_out.at<OutPixeType>(i, j) = static_cast<OutPixeType>(value);
    }
  }
}

#endif