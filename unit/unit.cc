#include "unit.h"

namespace MY_IMG {

const double PI = acos(-1);

void Rgb2Gray(const IMG_Mat &img, IMG_Mat &dimg, const std::vector<double> &w) {
  int width = img.cols;
  int height = img.rows;
  dimg = IMG_Mat(img.size(), CV_8UC1);

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      const cv::Vec3b &pixel = img.at<cv::Vec3b>(i, j);

      dimg.at<uint8_t>(i, j) =
          w[0] * pixel[0] + w[1] * pixel[1] + w[2] * pixel[2];
    }
  }
}

void HistogramEqualization(const IMG_Mat &input_img, IMG_Mat &output_img) {
  auto height = input_img.rows;
  auto width = input_img.cols;

  output_img = IMG_Mat(input_img.size(), input_img.type());

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
  return 1.0 / pow(2 * PI * sigma * sigma, n) * exp(-x / (2 * sigma * sigma));
}

void CreateGaussBlurFilter(IMG_Mat &filter, const double &sigma, int radim) {
  if (radim == -1) {
    int i = 0;
    double val = gaussFunction(sigma, 0, 0.5);
    while (gaussFunction(sigma, i * i, 0.5) > 0.01 * val) {
      i++;
    }
    radim = i * 2 + 1;
  }

  filter = IMG_Mat(radim, 1, CV_64F);

  for (int i = 0; i <= radim / 2; i++) {
    filter.at<double>(i + radim / 2) = filter.at<double>(radim / 2 - i) =
        gaussFunction(sigma, i * i, 0.5);
  }

  filter /= cv::sum(filter)[0];
}
// 生成二维高斯滤波器
void CreateGaussBlurFilter(IMG_Mat &filter, const double &sigma, int radim_h,
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

  filter = IMG_Mat(radim_h + 1, radim_w + 1, CV_64F);
  double sum = 0;
  for (int i = 0; i <= radim_h / 2; i++) {
    for (int j = 0; j <= radim_w / 2; j++) {
      filter.at<double>(i + radim_h / 2, j + radim_w / 2) =
          filter.at<double>(radim_h / 2 - i, radim_w / 2 - j) =
              filter.at<double>(radim_h / 2 - i, radim_w / 2 + j) =
                  filter.at<double>(radim_h / 2 + i, radim_w / 2 - j) =
                      gaussFunction(sigma, i * i + j * j, 1);
    }
  }
  for (int i = 0; i < radim_h + 1; i++) {
    for (int j = 0; j < radim_w + 1; j++) {
      sum += filter.at<double>(i, j);
    }
  }
  filter /= sum;
}
template <typename PixeType>
void filter2d_split(const IMG_Mat &img, IMG_Mat &blur_img,
                    const IMG_Mat &blur_filter) {
  blur_img = img.clone();
  int height = img.rows;
  int width = img.cols;
  for (int i = 0; i < height; i++) {
    std::vector<double> tmp;
    tmp.clear();
    for (int j = 0; j < width; j++) {
      double val = 0.0;
      for (int k = 0; k < blur_filter.rows; k++) {
        int x = j + k - blur_filter.rows / 2;
        if (x < 0) {
          x = -x;
        }
        if (x >= width) {
          x = 2 * width - x - 1;
        }
        val += blur_img.at<PixeType>(i, x) * blur_filter.at<double>(k);
      }
      tmp.push_back(val);
    }
    for (int j = 0; j < width; j++) {
      blur_img.at<PixeType>(i, j) = static_cast<PixeType>(tmp[j]);
    }
  }
  for (int i = 0; i < width; i++) {
    std::vector<double> tmp;
    tmp.clear();
    for (int j = 0; j < height; j++) {
      double val = 0.0;
      for (int k = 0; k < blur_filter.rows; k++) {
        int x = j + k - blur_filter.rows / 2;
        if (x < 0) {
          x = -x;
        }
        if (x >= height) {
          x = 2 * height - x - 1;
        }
        val += blur_img.at<PixeType>(x, i) * blur_filter.at<double>(k);
      }
      tmp.push_back(val);
    }
    for (int j = 0; j < height; j++) {
      blur_img.at<PixeType>(j, i) = static_cast<PixeType>(tmp[j]);
    }
  }
}
void filter2d_(const IMG_Mat &input_img, IMG_Mat &output_img,
               const IMG_Mat &filter) {
  int height = input_img.rows;
  int width = input_img.cols;
  int filter_height = filter.rows;
  int filter_width = filter.cols;

  IMG_Mat tmp_buffer = IMG_Mat(height, width, CV_8UC1);

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
  tmp_buffer.release();
}
void GaussBlur(const IMG_Mat &src, IMG_Mat &dst, const double &sigma,
               const double &radim) {
  IMG_Mat filter;
  CreateGaussBlurFilter(filter, sigma, radim);
  if (src.channels() == 1) {
    if (src.type() == CV_32F) {
      filter2d_split<float>(src, dst, filter);
    } else if (src.type() == CV_64F) {
      filter2d_split<double>(src, dst, filter);
    } else if (src.type() == CV_8U) {
      filter2d_split<uint8_t>(src, dst, filter);
    } else if (src.type() == CV_16U) {
      filter2d_split<uint16_t>(src, dst, filter);
    } else if (src.type() == CV_32S) {
      filter2d_split<int32_t>(src, dst, filter);
    } else {
      ASSERT(false, "GaussBlur: unsupported type");
    }
  } else {
    IMG_Mat tmp_buffer[src.channels()];
    cv::split(src, tmp_buffer);
    for (int i = 0; i < src.channels(); i++) {
      if (src.type() == CV_32FC3) {
        filter2d_split<float>(tmp_buffer[i], tmp_buffer[i], filter);
      } else if (src.type() == CV_64FC3) {
        filter2d_split<double>(tmp_buffer[i], tmp_buffer[i], filter);
      } else if (src.type() == CV_8UC3) {
        filter2d_split<uint8_t>(tmp_buffer[i], tmp_buffer[i], filter);
      } else if (src.type() == CV_16UC3) {
        filter2d_split<uint16_t>(tmp_buffer[i], tmp_buffer[i], filter);
      } else {
        ASSERT(false, "GaussBlur: unsupported type");
      }
    }
    cv::merge(tmp_buffer, src.channels(), dst);
    for (int i = 0; i < src.channels(); i++) {
      tmp_buffer[i].release();
    }
  }
}
void ImgFilter(const IMG_Mat &img, const IMG_Mat &filter, IMG_Mat &dimg,
               const bool &is_resverse) {
  int width = img.cols;
  int height = img.rows;
  int radim_h = static_cast<int>(filter.cols - 1);
  int radim_w = static_cast<int>(filter.rows - 1);
  assert(width > radim_w && height > radim_h);

  dimg = IMG_Mat(img.size(), img.type());

  // 获取图像类型和该类型变量的类型
  const int type = img.type();
  const int type_size = CV_MAT_CN(type);
  const int type_code = CV_MAT_DEPTH(type);

  LOG(INFO,"type_size: %d type_code: %d", type_size, type_code);

  // 分配内存
  IMG_Mat temp_img[type_size];
  for (int i = 0; i < type_size; i++) {
    temp_img[i] = IMG_Mat(height, width, CV_8UC1);
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
  for (int i = 0; i < type_size; i++) {
    temp_img[i].release();
  }
}
IMG_Mat ConvertComplexMat2doubleMat(const IMG_Mat &img) {
  int height = img.rows;
  int width = img.cols;
  IMG_Mat result = IMG_Mat(height, width, CV_64FC1);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      result.at<double>(i, j) = std::abs(img.at<std::complex<double>>(i, j));
    }
  }
  return result;
}

Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
ConvertMat2Eigen(const IMG_Mat &img) {
  int height = img.rows;
  int width = img.cols;
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> result(
      height, width);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      result(i, j) = img.at<std::complex<double>>(i, j);
    }
  }
  // LOG(INFO,"Eigen: (%ld,%ld)", result.rows(), result.cols());
  // LOG(INFO,"Mat: (%d,%d)", img.rows, img.cols);
  return result;
}
IMG_Mat ConvertEigen2Mat(
    const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
        &img) {
  int height = img.rows();
  int width = img.cols();
  IMG_Mat result = IMG_Mat(height, width, CV_64FC2);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      result.at<std::complex<double>>(i, j) = img(i, j);
    }
  }
  return result;
}
// 设置图像偏移n/2，m/2
void _fftShift(
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &img) {
  int height = img.rows();
  int width = img.cols();
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      img(i, j) *= pow(-1, i + j);
    }
  }
}
void _fftShift(IMG_Mat &img) {
  int height = img.rows;
  int width = img.cols;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      img.at<std::complex<double>>(i, j) *= pow(-1, i + j);
    }
  }
}
void _dft_core(
    const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
        &img,
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &dimg,
    std::function<std::complex<double>(double, double, double)> kernel) {
  int height = img.rows();
  int width = img.cols();
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> dft_mat_w(
      width, width);
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> dft_mat_h(
      height, height);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      dft_mat_w(i, j) = kernel(i, j, width);
    }
  }
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < height; j++) {
      dft_mat_h(i, j) = kernel(i, j, height);
    }
  }
  dimg = dft_mat_h * img * dft_mat_w;
  dimg /= sqrt(width * height);
}

void _fft_core(std::shared_ptr<std::complex<double>[]> src, int lim,
               const bool &is_fft) {
  int len = 0;
  while ((1 << len) < lim) {
    len++;
  }
  std::vector<int> rev(lim, 0);
  for (int i = 0; i < lim; i++) {
    rev.at(i) = (rev.at(i >> 1) >> 1) | ((i & 1) << (len - 1));
  }

  for (int i = 0; i < lim; i++) {
    if (i < rev.at(i)) {
      std::swap(src[i], src[rev.at(i)]);
    }
  }
  int opt = (is_fft ? -1 : 1);
  for (int m = 1; m <= lim; m <<= 1) {
    std::complex<double> wn(cos(2.0 * M_PI / m), opt * sin(2.0 * M_PI / m));
    for (int i = 0; i < lim; i += m) {
      std::complex<double> w(1, 0);
      for (int j = 0; j < (m >> 1); j++, w = w * wn) {
        ASSERT(i + j + (m >> 1) < lim, "i+j+(m >> 1) = %d,lim:%d",
               i + j + (m >> 1), lim);
        std::complex<double> u(0, 0), t(0, 0);
        u = src[i + j];
        t = w * src[i + j + (m >> 1)];
        src[i + j] = u + t, src[i + j + (m >> 1)] = u - t;
      }
    }
  }
}
void _fft2D(const IMG_Mat &img, IMG_Mat &dft_img, const bool &is_fft) {
  int height = img.rows;
  int width = img.cols;
  int lim_height = 1, lim_width = 1;
  while (lim_height < height) {
    lim_height <<= 1;
  }
  while (lim_width < width) {
    lim_width <<= 1;
  }
  dft_img = IMG_Mat(lim_height, lim_width, CV_64FC2);
  std::shared_ptr<std::complex<double>[]> tmp(
      new std::complex<double>[lim_width]());
  // for (int i = 0;i < lim_width;i++) {
  //   tmp[i] = std::complex<double>(0, 0);
  // }
  // LOG(INFO,"width : %d,size : %ld", width, tmp.size());
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      tmp[j] = img.at<std::complex<double>>(i, j);
    }
    for (int j = width; j < lim_width; j++) {
      tmp[j] = std::complex<double>(0, 0);
    }
    _fft_core(tmp,lim_width, is_fft);
    for (int j = 0; j < lim_width; j++) {
      dft_img.at<std::complex<double>>(i, j) = tmp[j] / sqrt(lim_width);
    }
  }
  for (int i = height; i < lim_height; i++) {
    for (int j = 0; j < lim_width; j++) {
      dft_img.at<std::complex<double>>(i, j) = std::complex<double>(0, 0);
    }
  }
  tmp.reset(new std::complex<double>[lim_height]());
  // LOG(INFO,"height : %d,size : %ld", height, tmp.size());
  for (int j = 0; j < lim_width; j++) {
    for (int i = 0; i < lim_height; i++) {
      tmp[i] = dft_img.at<std::complex<double>>(i, j);
    }
    _fft_core(tmp, lim_height, is_fft);
    for (int i = 0; i < lim_height; i++) {
      dft_img.at<std::complex<double>>(i, j) = tmp[i] / sqrt(lim_height);
    }
  }
  tmp.reset();
}
void FFT2D(const IMG_Mat &img, IMG_Mat &dft_img) {
  IMG_Mat temp_img;
  if (temp_img.type() == CV_8U) {
    temp_img = ConvertSingleChannelMat2ComplexMat<uint8_t>(img);
  } else if (temp_img.type() == CV_8S) {
    temp_img = ConvertSingleChannelMat2ComplexMat<int8_t>(img);
  } else if (temp_img.type() == CV_32F) {
    temp_img = ConvertSingleChannelMat2ComplexMat<float>(img);
  } else if (temp_img.type() == CV_64F) {
    temp_img = ConvertSingleChannelMat2ComplexMat<double>(img);
  } else {
    ASSERT(false, "Unsupported type");
  }
  _fftShift(temp_img);
  _fft2D(temp_img, dft_img, true);
  temp_img.release();
}
void IFFT2D(const IMG_Mat &img, IMG_Mat &dft_img) {
  IMG_Mat temp_img;
  _fft2D(img, temp_img, false);
  _fftShift(temp_img);
  int height = img.rows;
  int width = img.cols;
  dft_img = IMG_Mat(img.size(), CV_32F);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      float value =
          static_cast<float>(temp_img.at<std::complex<double>>(i, j).real());
      dft_img.at<float>(i, j) = static_cast<float>(value);
    }
  }
  temp_img.release();
}
void DFT(const IMG_Mat &img, IMG_Mat &dft_img) {
  // 将图像转换为复数矩阵
  IMG_Mat temp_img = ConvertSingleChannelMat2ComplexMat<uint8_t>(img);
  // LOG(INFO,"temp_img size(fftshift): (%ld,%ld)", temp_img.rows, temp_img.cols);
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
      eigen_img = ConvertMat2Eigen(temp_img);
  _fftShift(eigen_img);
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> dft_mat;
  _dft_core(eigen_img, dft_mat, [](double x, double u, double N) {
    return std::exp(std::complex<double>(0, -2 * M_PI * x * u / N));
  });
  // 转换为IMG_Mat
  dft_img = ConvertEigen2Mat(dft_mat);
  // LOG(INFO,"dft_img size: (%d,%d)", dft_img.rows, dft_img.cols);
  temp_img.release();
}

void IDFT(const IMG_Mat &dft_img, IMG_Mat &idft_img) {
  int height = dft_img.rows;
  int width = dft_img.cols;
  // 将图像转换为复数矩阵
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
      eigen_img = ConvertMat2Eigen(dft_img);
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> idft_mat;
  _dft_core(eigen_img, idft_mat, [](double x, double u, double N) {
    return std::exp(std::complex<double>(0, 2 * M_PI * x * u / N));
  });
  _fftShift(idft_mat);
  // 转换为IMG_Mat
  idft_img = IMG_Mat(height, width, CV_8UC1);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int value = static_cast<int>(round(idft_mat(i, j).real()));
      if (value < 0)
        value = 0;
      if (value > 255)
        value = 255;
      idft_img.at<uint8_t>(i, j) = static_cast<uint8_t>(value);
    }
  }
}
IMG_Mat GrayCorrosion(const IMG_Mat &src, const IMG_Mat &struct_element) {
  int height = src.rows;
  int width = src.cols;
  int struct_height = struct_element.rows;
  int struct_width = struct_element.cols;
  IMG_Mat dst = IMG_Mat(height, width, CV_8UC1);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int sum = 1e9;
      for (int m = 0; m < struct_height; m++) {
        for (int n = 0; n < struct_width; n++) {
          if (i + m > height || j + n > width) {
            continue;
          }
          sum = std::min(
              sum, static_cast<int>(src.at<uint8_t>(i + m, j + n)) -
                       static_cast<int>(struct_element.at<uint8_t>(m, n)));
        }
      }
      if (sum < 0) {
        sum = 0;
      } else if (sum > 255) {
        sum = 255;
      }
      dst.at<uint8_t>(i, j) = static_cast<uint8_t>(sum);
    }
  }
  return dst;
}
IMG_Mat GrayExpansion(const IMG_Mat &src, const IMG_Mat &struct_element) {
  int height = src.rows;
  int width = src.cols;
  int struct_height = struct_element.rows;
  int struct_width = struct_element.cols;
  IMG_Mat dst = IMG_Mat(height, width, CV_8UC1);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int sum = -1e9;
      for (int m = 0; m < struct_height; m++) {
        for (int n = 0; n < struct_width; n++) {
          if (i - m < 0 || j - n < 0) {
            continue;
          }
          sum = std::max(
              sum, static_cast<int>(src.at<uint8_t>(i - m, j - n)) +
                       static_cast<int>(struct_element.at<uint8_t>(m, n)));
        }
      }
      if (sum < 0) {
        sum = 0;
      } else if (sum > 255) {
        sum = 255;
      }
      dst.at<uint8_t>(i, j) = static_cast<uint8_t>(sum);
    }
  }
  return dst;
}
IMG_Mat GrayOpening(const IMG_Mat &src, const IMG_Mat &struct_element) {
  return GrayExpansion(GrayCorrosion(src, struct_element), struct_element);
}
IMG_Mat GrayClosing(const IMG_Mat &src, const IMG_Mat &struct_element) {
  return GrayCorrosion(GrayExpansion(src, struct_element), struct_element);
}
} // namespace MY_IMG