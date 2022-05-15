#include "image_operator.h"
#include "unit.h"

namespace MY_IMG {
double _S(double sigma) { return 0.001 * exp(-2 * sigma + 15); }
void ImageGaussianFilter(const IMG_Mat &img, IMG_Mat &img_out,
                         const double &sigma) {
  int channel = img.channels();
  IMG_Mat img_tmp;
  if (channel == 3) {
    Rgb2Gray(img, img_tmp);
  } else {
    img_tmp = img.clone();
  }
  // cv::imshow("img", img_tmp);
  IMG_Mat img_fft;
  FFT2D(img_tmp, img_fft);
  // cv::imshow("img_fft", ConvertSingleChannelMat2Uint8Mat<double>(
  //                           ConvertComplexMat2doubleMat(img_fft)));
  IMG_Mat img_tmp_out = IMG_Mat::zeros(img_fft.size(), img_fft.type());
  IMG_Mat img_ift;
  LOG("-> sigma: %f", _S(sigma));
  GaussianFilter filter(std::make_pair(img_fft.rows / 2, img_fft.cols / 2),
                        _S(sigma));
  // LOG("img_fft.size:%d,%d", img_fft.cols, img_fft.rows);
  for (int j = 0; j < img_fft.rows; ++j) {
    for (int k = 0; k < img_fft.cols; ++k) {
      img_tmp_out.at<std::complex<double>>(j, k) =
          filter.LowPassFilter(j, k) * img_fft.at<std::complex<double>>(j, k);
    }
  }
  IFFT2D(img_tmp_out, img_ift);
  // cv::imshow("img_ift",
  // ConvertSingleChannelMat2Uint8Mat<float>(img_tmp_out));
  // LOG("img_ift.size:%d,%d", img_ift.cols, img_ift.rows);
  img_out = img_ift(cv::Range(0, img_tmp.rows), cv::Range(0, img_tmp.cols));
  // cv::waitKey(0);
  // 释放Mat的内存
  img_ift.release();
  img_tmp_out.release();
  img_tmp.release();
}
/**
 * @brief 创建基于弧度的旋转矩阵R
 * @param[in] angle 弧度
 * @return R
 * @note R = [cos(angle),-sin(angle);sin(angle),cos(angle)]
 */
Eigen::Matrix2d rotate_matrix(const double &angle) {
  Eigen::Matrix2d R;
  R << cos(angle), -sin(angle), sin(angle), cos(angle);
  return R;
}
template <typename PixeType>
void _img_change(const IMG_Mat &src, IMG_Mat &out, const double &zoom,
                 const double &angle) {
  if (src.channels() != 1) {
    ASSERT(false, "ImageChange: src must be gray image");
  }
  if (zoom < 0) {
    ASSERT(false, "ImageChange: zoom must be positive");
  }
  Eigen::Matrix2d R = rotate_matrix(angle) * zoom;

  Eigen::Matrix3d T = Eigen::Matrix3d::Zero();
  Eigen::Vector2d center = Eigen::Vector2d::Zero();
  Eigen::Vector2d dst_center = Eigen::Vector2d::Zero();

  // 获取原图像中心点坐标
  center << src.rows / 2, src.cols / 2;

  // 获取旋转后图像的范围
  double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
  // 左上角点范围
  Eigen::Vector2d tmp_vec = R * Eigen::Vector2d(-center[0], center[1]);
  min_x = std::min(min_x, tmp_vec(0));
  min_y = std::min(min_y, tmp_vec(1));
  max_x = std::max(max_x, tmp_vec(0));
  max_y = std::max(max_y, tmp_vec(1));
  // 右上角点范围
  tmp_vec = R * Eigen::Vector2d(center[0], center[1]);
  min_x = std::min(min_x, tmp_vec(0));
  min_y = std::min(min_y, tmp_vec(1));
  max_x = std::max(max_x, tmp_vec(0));
  max_y = std::max(max_y, tmp_vec(1));
  // 左下角点范围
  tmp_vec = R * Eigen::Vector2d(-center[0], -center[1]);
  min_x = std::min(min_x, tmp_vec(0));
  min_y = std::min(min_y, tmp_vec(1));
  max_x = std::max(max_x, tmp_vec(0));
  max_y = std::max(max_y, tmp_vec(1));
  // 右下角点范围
  tmp_vec = R * Eigen::Vector2d(center[0], -center[1]);
  min_x = std::min(min_x, tmp_vec(0));
  min_y = std::min(min_y, tmp_vec(1));
  max_x = std::max(max_x, tmp_vec(0));
  max_y = std::max(max_y, tmp_vec(1));

  // 计算旋转后图像的中心点坐标
  dst_center << (max_x - min_x) / 2, (max_y - min_y) / 2;
  LOG("dst_center: %lf,%lf", dst_center[0], dst_center[1]);
  // 计算从dst到src的T
  T.block<2, 2>(0, 0) = R.inverse();
  T.block<2, 1>(0, 2) = (center - R.inverse() * dst_center);
  T(2, 2) = 1;

  IMG_Mat dst = cv::Mat(dst_center[0] * 2, dst_center[1] * 2, src.type());
  LOG("dst: %d,%d", dst.rows, dst.cols);
  auto func = [&src](Eigen::Vector3d pose) -> PixeType {
    int x = static_cast<int>(pose[0]), y = static_cast<int>(pose[1]);
    if (x < 0 || y < 0 || x >= src.rows || y >= src.cols) {
      return 0;
    }
    return src.at<PixeType>(x, y);
  };
  // 算范围，改变图像框大小了，（方案2）
  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      Eigen::Vector3d pose = Eigen::Vector3d::Zero();
      pose << i, j, 1;
      dst.at<PixeType>(i, j) = func(T * pose);
    }
  }

  out = dst.clone();
  dst.release();
}

void ImageChange(const IMG_Mat &src, IMG_Mat &dst, const double &zoom,
                 const double &angle) {
  if (src.type() == CV_8U) {
    _img_change<uchar>(src, dst, zoom, angle);
  } else if (src.type() == CV_8S) {
    _img_change<char>(src, dst, zoom, angle);
  } else if (src.type() == CV_32F) {
    _img_change<float>(src, dst, zoom, angle);
  } else if (src.type() == CV_64F) {
    _img_change<double>(src, dst, zoom, angle);
  } else {
    ASSERT(false, "ImageChange: unsupported type");
  }
}

void DrawPoints(const IMG_Mat &img,
                const std::vector<std::shared_ptr<KeyPoint>> &keypoints,
                IMG_Mat &img_out) {
  img_out = img.clone();
  for (const auto &p : keypoints) {
    cv::circle(img_out, cv::Point(p->y, p->x), 3, cv::Scalar(0, 255, 0), 1);
    cv::arrowedLine(img_out, cv::Point(p->y, p->x),
                    cv::Point(p->y + sin(p->angle) * p->size,
                              p->x + cos(p->angle) * p->size),
                    cv::Scalar(255, 0, 0), 1, 8, 0, 0.1);
  }
}
void DrawMatch(
    const IMG_Mat &img1, const IMG_Mat &img2,
    const std::vector<std::pair<std::shared_ptr<KeyPoint>,
                                std::vector<std::shared_ptr<KeyPoint>>>>
        &match_result,
    IMG_Mat &img_out) {
  ASSERT(img1.type() == img2.type(), "img1 and img2 type must be same");
  cv::hconcat(img1, img2, img_out);

  for (const auto &p : match_result) {
    cv::circle(img_out, cv::Point(p.first->y, p.first->x), 3,
               cv::Scalar(0, 0, 255), 1);
    cv::circle(img_out, cv::Point(p.second[0]->y + img1.cols, p.second[0]->x), 3,
               cv::Scalar(0, 0, 255), 1);
    cv::line(img_out, cv::Point(p.first->y, p.first->x),
             cv::Point(p.second[0]->y + img1.cols, p.second[0]->x), cv::Scalar(0, 255, 0),
             1);
  }
}
}; // namespace MY_IMG
