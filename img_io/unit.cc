#include "unit.h"
#include <cmath>
#include <complex>

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
      if(sum < 0) {
        sum = 0;
      }
      if(sum > 255) {
        sum = 255;
      }
      tmp_buffer.at<uint8_t>(i, j) = static_cast<uint8_t>(sum);
    }
  }
  // 将结果拷贝到输出图像
  tmp_buffer.copyTo(output_img);
}

void ImgFilter(const cv::Mat &img, const cv::Mat &filter, cv::Mat &dimg,const bool &is_resverse) {
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
  
  LOG("type_size: %d type_code: %d",type_size,type_code);

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
    if(is_resverse) {
      temp_img[i] = 255 - temp_img[i];
    }
  }

  // 合并通道
  cv::merge(temp_img, type_size, dimg);
}
struct F{
  const cv::Mat &img;
  cv::Mat &dimg;
  bool is_dft;
  // 构造函数
  F(const cv::Mat &img, cv::Mat &dimg,const bool is_dft = true) : img(img), dimg(dimg),is_dft(is_dft) {}
  // 滤波函数
  double operator()(int u,int v) const {
    int height = img.rows;
    int width = img.cols;

    std::complex<double> sum(0, 0);
    double coef = -2;
    if(!is_dft) {
      coef = 2;
    }
    for(int i = 0;i < height;i++) {
      for(int j = 0;j < width;j++) {
        sum += std::complex<double>(img.at<uint8_t>(i, j) , 0) * std::exp(std::complex<double>(0, coef * M_PI * (i * u * 1.0 / height + j * v * 1.0 / width)));
      }
    }
    sum /= height * width;
    double mag = std::abs(sum);
    return mag;
  }
};
void dft_(const cv::Mat &src,cv::Mat &dimg){
  int width = src.cols;
  int height = src.rows;

  cv::Mat tmp_img = cv::Mat(src.size(), CV_64FC1);
  F f(src, tmp_img);
  #pragma omp parallel for schedule(dynamic)
  for(int i = 0;i < height;i++) {
    #pragma omp parallel for schedule(dynamic)
    for(int j = 0;j < width;j++) {
      tmp_img.at<double>(i,j) = f(i + (height - 1)/2, j + (width - 1)/2);
    }
  }
  tmp_img.copyTo(dimg);
}
// 实现傅里叶变换
void DFT(const cv::Mat &src, cv::Mat &dimg) {
  int width = src.cols;
  int height = src.rows;
  int radim_h = static_cast<int>(src.cols - 1);
  int radim_w = static_cast<int>(src.rows - 1);

  dimg = cv::Mat(src.size(), CV_64FC1);

  // 获取图像类型和该类型变量的类型
  const int type = src.type();
  const int type_size = CV_MAT_CN(type);
  const int type_code = CV_MAT_DEPTH(type);

  // 分配内存
  cv::Mat temp_img[type_size];
  for (int i = 0; i < type_size; i++) {
    temp_img[i] = cv::Mat(height, width, CV_64FC1);
  }

  // 分离通道
  cv::split(src, temp_img);

  // 对每个通道进行傅里叶变换
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < type_size; i++) {
    dft_(temp_img[i], temp_img[i]);
  }

  // 合并通道
  cv::merge(temp_img, type_size, dimg);
}
void idft_(const cv::Mat &src,cv::Mat &dimg){
  int width = src.cols;
  int height = src.rows;

  cv::Mat tmp_img = cv::Mat(src.size(), CV_64FC1);
  F f(src, tmp_img,false);
  #pragma omp parallel for schedule(dynamic)
  for(int i = 0;i < height;i++) {
    #pragma omp parallel for schedule(dynamic)
    for(int j = 0;j < width;j++) {
      tmp_img.at<double>(i,j) = f(i + (height - 1)/2, j + (width - 1)/2);
    }
  }
  tmp_img.copyTo(dimg);
}
void IDFT(const cv::Mat &src, cv::Mat &dimg){
  int width = src.cols;
  int height = src.rows;
  int radim_h = static_cast<int>(src.cols - 1);
  int radim_w = static_cast<int>(src.rows - 1);

  dimg = cv::Mat(src.size(), CV_64FC1);

  // 获取图像类型和该类型变量的类型
  const int type = src.type();
  const int type_size = CV_MAT_CN(type);
  const int type_code = CV_MAT_DEPTH(type);

  // 分配内存
  cv::Mat temp_img[type_size];
  for (int i = 0; i < type_size; i++) {
    temp_img[i] = cv::Mat(height, width, CV_64FC1);
  }

  // 分离通道
  cv::split(src, temp_img);

  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < type_size; i++) {
    idft_(temp_img[i], temp_img[i]);
  }

  // 合并通道
  cv::merge(temp_img, type_size, dimg);
}
} // namespace MY_IMG