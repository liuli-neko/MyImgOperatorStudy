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
void ImageGaussianFilter(const IMG_Mat &img, IMG_Mat &img_out,
                         const double &sigma);
/**
 * @brief 图像旋转
 * @param[in] src 原图像
 * @param[out] dst 旋转后的图像
 * @param[in] zoom 缩放比例
 * @param[in] angle 旋转角度(弧度)
 */
void ImageChange(const IMG_Mat &img, IMG_Mat &img_out, const double &zoom = 2,
                 const double &angle = 0);
void DrawPoints(const IMG_Mat &img, const std::vector<KeyPoint> &keypoints,
                IMG_Mat &img_out);

template <typename InPixeType, typename OutPixeType>
void ConvertTo(const IMG_Mat &img, IMG_Mat &img_out,std::function<OutPixeType(const InPixeType&)> &func);
} // namespace MY_IMG
template <typename InPixeType, typename OutPixeType>
void MY_IMG::ConvertTo(const IMG_Mat &img, IMG_Mat &img_out,const std::function<OutPixeType(const InPixeType&)> &func) {
  ASSERT(img_out.empty() == false, "ConvertTo: img is empty");
  ASSERT(img.cols <= img_out.cols && img.rows <= img_out.rows,
         "ConvertTo: img size[%d %d] is not equal to img_out[%d %d]", img.cols,
         img.rows, img_out.cols, img_out.rows);
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      img_out.at<OutPixeType>(i, j) = func(img.at<InPixeType>(i, j));
    }
  }
}

#endif