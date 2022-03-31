#ifndef IMG_IO_UNIT_H_
#define IMG_IO_UNIT_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

namespace MY_IMG {

#define GRAY_MAX 255
#define LOG(format, ...) printf("[%s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)

void HistogramEqualization(const cv::Mat &input_img, cv::Mat &output_img);
void CreateGaussBlurFilter(cv::Mat &filter, const double &sigma, int radim);
void CreateGaussBlurFilter(cv::Mat &filter, const double &sigma, int radim_h,
                           int radim_w);
void ImgFilter(const cv::Mat &img, const cv::Mat &filter, cv::Mat &dimg,const bool &is_resverse = false);
// 定义傅里叶变换的函数声明
void DFT(const cv::Mat &src, cv::Mat &dst);
} // namespace MY_IMG

#endif