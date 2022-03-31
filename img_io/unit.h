#ifndef IMG_IO_UNIT_H_
#define IMG_IO_UNIT_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

namespace MY_IMG
{

#define GRAY_MAX 255
#define LOG(format, ...) printf("[%s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)

    const cv::Mat H1 = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    const cv::Mat H2 = (cv::Mat_<double>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
    const cv::Mat H3 = (cv::Mat_<double>(3, 3) << 1, -2, 1, -2, 4, -2, 1, -2, 1);
    const cv::Mat H4 = (cv::Mat_<double>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
    const cv::Mat H5 = (cv::Mat_<double>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
    const cv::Mat H6 = (cv::Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    const cv::Mat H7 = (cv::Mat_<double>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
    const cv::Mat H8 = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -5, 1, 0, 1, 0);
    const cv::Mat H9 = (cv::Mat_<double>(3, 3) << 1, 1, 1, 1, -9, 1, 1, 1, 1);

    void HistogramEqualization(const cv::Mat &input_img, cv::Mat &output_img);
    void CreateGaussBlurFilter(cv::Mat &filter, const double &sigma, int radim = -1);
    void CreateGaussBlurFilter(cv::Mat &filter, const double &sigma, int radim_h,
                               int radim_w);
    void ImgFilter(const cv::Mat &img, const cv::Mat &filter, cv::Mat &dimg, const bool &is_resverse = false);
    // 定义傅里叶变换的函数声明
    void DFT(const cv::Mat &src, cv::Mat &dst);
    void IDFT(const cv::Mat &src, cv::Mat &dst);
} // namespace MY_IMG

#endif