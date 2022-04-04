#include "unit.h"
#include <cmath>
#include <complex>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace MY_IMG {

const double PI = acos(-1);

void HistogramEqualization(const cv::Mat &input_img, cv::Mat &output_img) {
  auto height = input_img.rows;
  auto width = input_img.cols;

  output_img = cv::Mat(input_img.size(), input_img.type());

  // 统计灰度分布
  uint32_t n[GRAY_MAX + 1];
  memset(n, 0, sizeof(n));
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      n[input_img.at<uint8_t>(i, j)]++;
    }
  }

  // 归一化灰度，计算概率
  double p[GRAY_MAX + 1];
  for (int i = 0; i < GRAY_MAX + 1; i++) {
    p[i] = static_cast<double>(n[i]) / static_cast<double>(height * width);
  }

  // 计算新图像
  double sum_p[GRAY_MAX + 1];
  sum_p[0] = p[0];
  for (int i = 1; i < GRAY_MAX + 1; i++) {
    sum_p[i] = p[i] + sum_p[i - 1];
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      output_img.at<uint8_t>(i, j) = static_cast<uint8_t>(
          round(sum_p[input_img.at<uint8_t>(i, j)] * GRAY_MAX));
    }
  }
}

double gaussFunction(const double sigma, const double x, const double n) {
  return 1.0 / pow(2 * PI, n) * exp(-x / (2 * sigma * sigma));
}

void CreateGaussBlurFilter(cv::Mat &filter, const double &sigma, int radim) {
  if (radim == -1) {
    int i = 0;
    double val = gaussFunction(sigma, 0, 0.5);
    while (gaussFunction(sigma, i * i, 0.5) > 0.01 * val) {
      i++;
    }
    radim = i * 2;
  }

  filter = cv::Mat(radim + 1, 1, CV_64F);

  for (int i = 0; i <= radim / 2; i++) {
    filter.at<double>(i + radim / 2) = filter.at<double>(radim / 2 - i) =
        gaussFunction(sigma, i * i, 0.5);
  }

  filter /= cv::sum(filter)[0];
}
// 生成二维高斯滤波器
void CreateGaussBlurFilter(cv::Mat &filter, const double &sigma, int radim_h,
                           int radim_w) {
  if (radim_h == -1) {
    int i = 0;
    double val = gaussFunction(sigma, 0, 1);
    while (gaussFunction(sigma, i * i + i * i, 1) > 0.01 * val) {
      i++;
    }
    radim_h = i * 2;
  }

  if (radim_w == -1) {
    radim_w = radim_h;
  }

  filter = cv::Mat(radim_h + 1, radim_w + 1, CV_64F);

  for (int i = 0; i <= radim_h / 2; i++) {
    for (int j = 0; j <= radim_w / 2; j++) {
      filter.at<double>(i + radim_h / 2, j + radim_w / 2) =
          filter.at<double>(radim_h / 2 - i, radim_w / 2 - j) =
              filter.at<double>(radim_h / 2 - i, radim_w / 2 + j) =
                  filter.at<double>(radim_h / 2 + i, radim_w / 2 - j) =
                      gaussFunction(sigma, i * i + j * j, 1);
    }
  }

  filter /= cv::sum(filter)[0];
}

void filter2d_(const cv::Mat &input_img, cv::Mat &output_img,
               const cv::Mat &filter) {
  int height = input_img.rows;
  int width = input_img.cols;
  int filter_height = filter.rows;
  int filter_width = filter.cols;

  cv::Mat tmp_buffer = cv::Mat(height, width, CV_8UC1);

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      double sum = 0;
      for (int m = 0; m < filter_height; m++) {
        for (int n = 0; n < filter_width; n++) {
          int x = i + m - filter_height / 2;
          int y = j + n - filter_width / 2;

          if (x < 0) {
            x = -x;
          }
          if (y < 0) {
            y = -y;
          }
          if (x >= height) {
            x = 2 * height - x - 1;
          }
          if (y >= width) {
            y = 2 * width - y - 1;
          }

          sum += input_img.at<uint8_t>(x, y) * filter.at<double>(m, n);
        }
      }
      if (sum < 0) {
        sum = 0;
      }
      if (sum > 255) {
        sum = 255;
      }
      tmp_buffer.at<uint8_t>(i, j) = static_cast<uint8_t>(sum);
    }
  }
  // 将结果拷贝到输出图像
  tmp_buffer.copyTo(output_img);
}

void ImgFilter(const cv::Mat &img, const cv::Mat &filter, cv::Mat &dimg,
               const bool &is_resverse) {
  int width = img.cols;
  int height = img.rows;
  int radim_h = static_cast<int>(filter.cols - 1);
  int radim_w = static_cast<int>(filter.rows - 1);
  assert(width > radim_w && height > radim_h);

  dimg = cv::Mat(img.size(), img.type());

  // 获取图像类型和该类型变量的类型
  const int type = img.type();
  const int type_size = CV_MAT_CN(type);
  const int type_code = CV_MAT_DEPTH(type);

  LOG("type_size: %d type_code: %d", type_size, type_code);

  // 分配内存
  cv::Mat temp_img[type_size];
  for (int i = 0; i < type_size; i++) {
    temp_img[i] = cv::Mat(height, width, CV_8UC1);
  }

  // 分离通道
  cv::split(img, temp_img);

// 对每个通道进行滤波
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < type_size; i++) {
    filter2d_(temp_img[i], temp_img[i], filter);
    if (is_resverse) {
      temp_img[i] = 255 - temp_img[i];
    }
  }

  // 合并通道
  cv::merge(temp_img, type_size, dimg);
}
cv::Mat ConvertComplexMat2doubleMat(const cv::Mat &img) {
  int height = img.rows;
  int width = img.cols;
  cv::Mat result = cv::Mat(height, width, CV_64FC1);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      result.at<double>(i, j) = std::abs(img.at<std::complex<double>>(i, j));
    }
  }
  return result;
}
cv::Mat ConvertDoubleMat2Uint8Mat(const cv::Mat &img) {
  int height = img.rows;
  int width = img.cols;
  cv::Mat result = cv::Mat(height, width, CV_8UC1);
  // 获取图像中最大值和最小值
  double min_value = 0;
  double max_value = 0;
  cv::minMaxLoc(img, &min_value, &max_value);
  // 对每个像素进行转换
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      result.at<uint8_t>(i, j) =
          static_cast<uint8_t>(255 * (img.at<double>(i, j) - min_value) /
                               (max_value - min_value));
    }
  }
  return result;
}
Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> ConvertMat2Eigen(
    const cv::Mat &img) {
  int height = img.rows;
  int width = img.cols;
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> result(height, width);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      result(i, j) = img.at<std::complex<double>>(i, j);
    }
  }
  return result;
}
cv::Mat ConvertEigen2Mat(const Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic> &img) {
  int height = img.rows();
  int width = img.cols();
  cv::Mat result = cv::Mat(height, width, CV_64FC2);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      result.at<std::complex<double>>(i, j) = img(i, j);
    }
  }
  return result;
}
void DFT(const cv::Mat &img, cv::Mat &dft_img) {
  int height = img.rows;
  int width = img.cols;
  // 将图像转换为复数矩阵
  cv::Mat temp_img = ConvertSingleChannelMat2ComplexMat<uint8_t>(img);
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> eigen_img = ConvertMat2Eigen(temp_img);
  
  // 创建傅里叶变换矩阵
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> dft_mat_w =
      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>::Zero(width, width);
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> dft_mat_h =
      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>::Zero(height, height);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      dft_mat_w(i, j) = std::exp(std::complex<double>(0, -2 * M_PI * i * j / width));
    }
  }
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < height; j++) {
      dft_mat_h(i, j) = std::exp(std::complex<double>(0, -2 * M_PI * i * j / height));
    }
  }
  // 傅里叶变换
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> dft_mat = dft_mat_w * eigen_img * dft_mat_h;
  dft_mat /= width * height;
  // 转换为cv::Mat
  dft_img = ConvertEigen2Mat(dft_mat);
}
void IDFT(const cv::Mat &dft_img, cv::Mat &idft_img) {
  int height = dft_img.rows;
  int width = dft_img.cols;
  // 将图像转换为复数矩阵
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> eigen_img = ConvertMat2Eigen(dft_img);
  // 创建傅里叶逆变换矩阵
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> idft_mat_w =
      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>::Zero(width, width);
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> idft_mat_h =
      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>::Zero(height, height);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      idft_mat_w(i, j) = std::exp(std::complex<double>(0, 2 * M_PI * i * j / width));
    }
  }
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < height; j++) {
      idft_mat_h(i, j) = std::exp(std::complex<double>(0, 2 * M_PI * i * j / height));
    }
  }
  // 傅里叶逆变换
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> idft_mat = idft_mat_w * eigen_img * idft_mat_h;
  idft_img   /= width * height;
  // 转换为cv::Mat
  idft_img = cv::Mat(height, width, CV_8UC1);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int value = static_cast<int>(idft_mat(i, j).real() + 0.5);
      if(value < 0) value = 0;
      if(value > 255) value = 255;
      idft_img.at<uint8_t>(i, j) = static_cast<uint8_t>(value);
    }
  }
}
} // namespace MY_IMG