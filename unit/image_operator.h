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
void ImageGaussianFilter(const IMG_Mat &img, std::vector<IMG_Mat> &img_out,const std::vector<double> &sigma);
void ImageChange(const IMG_Mat &img, IMG_Mat &img_out,const double &zoom = 2,const double &angle = 0);
void DrawPoints(const IMG_Mat &img, const std::vector<Point2d> &keypoints, IMG_Mat &img_out);
} // namespace MY_IMG

#endif