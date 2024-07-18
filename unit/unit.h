#ifndef IMG_OPERATOR_UNIT_UNTI_H_
#define IMG_OPERATOR_UNIT_UNTI_H_
#define USE_OPENCV
#define USE_EIGEN
#include "all.h"

const double PI = acos(-1.0);

namespace MY_IMG {
#define GRAY_MAX 255

// 拉普拉斯算子
// H1~H5用于图像特征提取
// H6~H9用于图像锐化
const IMG_Mat H1 = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
const IMG_Mat H2 = (cv::Mat_<double>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
const IMG_Mat H3 = (cv::Mat_<double>(3, 3) << 1, -2, 1, -2, 4, -2, 1, -2, 1);
const IMG_Mat H4 = (cv::Mat_<double>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
const IMG_Mat H5 = (cv::Mat_<double>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
const IMG_Mat H6 = (cv::Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
const IMG_Mat H7 = (cv::Mat_<double>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
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
    ButterworthFilter(std::pair<double, double> center_, double D0_ = 10, int n_ = 2, double W_ = 1.0)
        : D0(D0_), n(n_), W(W_), center(center_) {}
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
    GaussianFilter(std::pair<double, double> center_, double D0_ = 1.0) : D0(D0_), center(center_) {}
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
void Rgb2Gray(const IMG_Mat& src, IMG_Mat& dst, const std::vector<double>& w = {0.299, 0.587, 0.114});
/**
 * @brief 对图像进行直方图均衡化
 * @param src 原图像
 * @param dst 均衡化后的图像
 */
void HistogramEqualization(const IMG_Mat& src, IMG_Mat& dst);
/**
 * @brief 创建高斯滤波卷积核
 * @param filter 滤波核
 * @param sigma 标准差
 * @param radim 滤波核半径
 * @note 滤波核半径为-1时将自动计算
 */
void CreateGaussBlurFilter(IMG_Mat& filter, const double& sigma, int radim = -1);
void CreateGaussBlurFilter(IMG_Mat& filter, const double& sigma, int radim_h, int radim_w);
/**
 * @brief 对图像进行高斯滤波（分离）
 * @param src 原图像
 * @param dst 滤波后的图像
 * @param sigma 标准差
 * @param radim 滤波核半径(-1时将自动计算,基于经验上的最大影响范围)
 * @note 基于高斯滤波的特殊性的样子，应该不能通用
 */
void GaussBlur(const IMG_Mat& src, IMG_Mat& dst, const double& sigma, const double& radim = -1);
/**
 * @brief 对图像进行卷积
 * @param src 原图像
 * @param dst 卷积后的图像
 * @param filter 滤波核
 * @param is_resverse 是否反转卷积后的图像
 */
void ImgFilter(const IMG_Mat& img, const IMG_Mat& filter, IMG_Mat& dst, const bool& is_resverse = false);

// 转换图像存储格式
IMG_Mat ConvertComplexMat2doubleMat(const IMG_Mat& img);
template <typename T>
IMG_Mat ConvertSingleChannelMat2ComplexMat(const IMG_Mat& img);
template <typename PixeType>
IMG_Mat ConvertSingleChannelMat2Uint8Mat(const IMG_Mat& img, const bool& is_mapping = false);

// 定义傅里叶变换的函数声明
void DFT(const IMG_Mat& src, IMG_Mat& dst);
void IDFT(const IMG_Mat& src, IMG_Mat& dst);
void FFT2D(const IMG_Mat& src, IMG_Mat& dst);
void IFFT2D(const IMG_Mat& src, IMG_Mat& dst);

// 形态学操作
IMG_Mat GrayErode(const IMG_Mat& src, const IMG_Mat& struct_element);
IMG_Mat GrayDilate(const IMG_Mat& src, const IMG_Mat& struct_element);
IMG_Mat GrayOpening(const IMG_Mat& src, const IMG_Mat& struct_element);
IMG_Mat GrayClosing(const IMG_Mat& src, const IMG_Mat& struct_element);

} // namespace MY_IMG
template <typename T>
IMG_Mat MY_IMG::ConvertSingleChannelMat2ComplexMat(const IMG_Mat& img) {
    int height     = img.rows;
    int width      = img.cols;
    IMG_Mat result = IMG_Mat(height, width, CV_64FC2);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            result.at<std::complex<double>>(i, j) = std::complex<double>(static_cast<double>(img.at<T>(i, j)), 0);
        }
    }
    return result;
}
template <typename PixeType>
IMG_Mat MY_IMG::ConvertSingleChannelMat2Uint8Mat(const IMG_Mat& img, const bool& is_mapping) {
    int height     = img.rows;
    int width      = img.cols;
    IMG_Mat result = IMG_Mat(height, width, CV_8UC1);
    // 获取图像中最大值和最小值
    double min_value = 0;
    double max_value = 0;
    cv::minMaxLoc(img, &min_value, &max_value);
    // 对每个像素进行转换
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (is_mapping) {
                result.at<uint8_t>(i, j) =
                    static_cast<uint8_t>(255 * (img.at<PixeType>(i, j) - min_value) / (max_value - min_value));
            } else {
                if (img.at<PixeType>(i, j) < 0) {
                    result.at<uint8_t>(i, j) = 0;
                } else if (img.at<PixeType>(i, j) > 255) {
                    result.at<uint8_t>(i, j) = 255;
                } else {
                    result.at<uint8_t>(i, j) = static_cast<uint8_t>(img.at<PixeType>(i, j));
                }
            }
        }
    }
    return result;
}
#endif