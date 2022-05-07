#include "image_operator.h"
#include "unit.h"

namespace MY_IMG {

void ImageGaussianFilter(IMG_Mat &img, std::vector<IMG_Mat> &img_out,const std::vector<double> &sigma){
    int channel = img.channels();
    IMG_Mat img_tmp;
    if (channel == 3) {
        Rgb2Gray(img,img_tmp,{1/3.0,1/3.0,1/3.0});
    }
    else {
        img_tmp = img;
    }
    IMG_Mat img_fft;
    GaussianFilter filter(std::make_pair(img_tmp.cols/2,img_tmp.rows/2),sigma[0]);
    FFT2D(img_tmp,img_fft);
    for (int i = 0; i < sigma.size(); ++i) {
        IMG_Mat img_tmp_out;
        filter.D0 = sigma[i];
        for (int j = 0; j < img_fft.rows; ++j) {
            for (int k = 0; k < img_fft.cols; ++k) {
                img_tmp_out.at<std::complex<double>>(j,k) = filter.LowPassFilter(i,j) * img_fft.at<std::complex<double>>(j,k);
            }
        }
        IFFT2D(img_tmp_out,img_tmp_out);
        img_out.push_back(img_tmp_out(cv::Range(0,img.rows),cv::Range(0,img.cols)));
        // 释放Mat的内存
        img_tmp_out.release();
    }
    img_tmp.release();
}
/**
 * @brief 创建基于弧度的旋转矩阵R
 * @param[in] angle 弧度
 * @return R
 * @note R = [cos(angle),-sin(angle);sin(angle),cos(angle)]
 */
Eigen::Matrix2d rotate_matrix(const double &angle){
  Eigen::Matrix2d R;
  R << cos(angle),-sin(angle),
       sin(angle),cos(angle);
  return R;
}
/**
 * @brief 图像旋转
 * @param[in] src 原图像
 * @param[out] dst 旋转后的图像
 * @param[in] zoom 缩放比例
 * @param[in] angle 旋转角度(弧度)
 */
void ImageChange(IMG_Mat &src, IMG_Mat &dst,const double &zoom,const double &angle){
  Eigen::Matrix2d R = rotate_matrix(angle) * zoom;

  Eigen::Matrix3d T = Eigen::Matrix3d::Zero();
  Eigen::Vector2d center = Eigen::Vector2d::Zero();
  Eigen::Vector2d dst_center = Eigen::Vector2d::Zero();

  // 获取原图像中心点坐标
  center << src.rows / 2,src.cols / 2;

  // 获取旋转后图像的范围
  double min_x = 1e9,max_x = -1e9,min_y = 1e9,max_y = -1e9;
  // 左上角点范围
  Eigen::Vector2d tmp_vec = R * Eigen::Vector2d(-center[0],center[1]);
  min_x = std::min(min_x,tmp_vec(0));
  min_y = std::min(min_y,tmp_vec(1));
  max_x = std::max(max_x,tmp_vec(0));
  max_y = std::max(max_y,tmp_vec(1));
  // 右上角点范围
  tmp_vec = R * Eigen::Vector2d(center[0],center[1]);
  min_x = std::min(min_x,tmp_vec(0));
  min_y = std::min(min_y,tmp_vec(1));
  max_x = std::max(max_x,tmp_vec(0));
  max_y = std::max(max_y,tmp_vec(1));
  // 左下角点范围
  tmp_vec = R * Eigen::Vector2d(-center[0],-center[1]);
  min_x = std::min(min_x,tmp_vec(0));
  min_y = std::min(min_y,tmp_vec(1));
  max_x = std::max(max_x,tmp_vec(0));
  max_y = std::max(max_y,tmp_vec(1));
  // 右下角点范围
  tmp_vec = R * Eigen::Vector2d(center[0],-center[1]);
  min_x = std::min(min_x,tmp_vec(0));
  min_y = std::min(min_y,tmp_vec(1));
  max_x = std::max(max_x,tmp_vec(0));
  max_y = std::max(max_y,tmp_vec(1));

  // 计算旋转后图像的中心点坐标
  dst_center << (max_x - min_x) / 2,(max_y - min_y) / 2;
  LOG("dst_center: %lf,%lf",dst_center[0],dst_center[1]);
  // 计算从dst到src的T
  T.block<2,2>(0,0) = R.inverse();
  T.block<2,1>(0,2) = (center - R.inverse() * dst_center);
  T(2,2) = 1;

  dst = cv::Mat(dst_center[0]*2,dst_center[1]*2,src.type());
  LOG("dst: %d,%d",dst.rows,dst.cols);
  auto func = [&src](Eigen::Vector3d pose) -> uint8_t{
    int x = static_cast<int>(pose[0]),y = static_cast<int>(pose[1]);
    if(x < 0 || y < 0 || x >= src.rows || y >= src.cols){
      return 0;
    }
    return src.at<uint8_t>(x,y);
  };
  // 算范围，改变图像框大小了，（方案2）
  for(int i = 0;i < dst.rows;i ++) {
    for(int j = 0;j < dst.cols;j ++) {
      Eigen::Vector3d pose = Eigen::Vector3d::Zero();
      pose << i,j,1;
      dst.at<uint8_t>(i,j) = func(T*pose);
    }
  }
}
}; // namespace MY_IMG
