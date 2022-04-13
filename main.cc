#include <iostream>
#include <unit.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

std::string img_path = "test.png";
std::string output = "rota_test.png";

Eigen::Matrix2d rotate_matrix(const double &angle){
  Eigen::Matrix2d R;
  R << cos(angle),-sin(angle),
       sin(angle),cos(angle);
  return R;
}

void rotate_img(const cv::Mat &src,cv::Mat &dst,double angle){
  angle -= static_cast<int>(angle/M_PI)*M_PI;
  Eigen::Matrix2d R = rotate_matrix(angle);
  std::cout << "R:" << std::endl;
  std::cout << R << std::endl;
  Eigen::Matrix3d T = Eigen::Matrix3d::Zero();
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d dst_center = Eigen::Vector3d::Zero();
  T.block<2,2>(0,0) = R;
  T(2,2) = 1;
  std::cout << "T:" << std::endl;
  std::cout << T << std::endl;
  // 获取原图像中心点坐标
  center << src.rows / 2,src.cols / 2,1;
  // 获取从dst到src的T
  Eigen::Matrix3d TT = T.inverse();
  TT.block<3,1>(0,2) = center;
  std::cout << "TT:" << std::endl;
  std::cout << TT << std::endl;
  dst = cv::Mat(src.size(),src.type());
  auto func = [&src](Eigen::Vector3d pose) -> uint8_t{
    int x = static_cast<int>(pose[0]),y = static_cast<int>(pose[1]);
    if(x < 0 || y < 0 || x > src.rows || y > src.cols){
      return 0;
    }
    return src.at<uint8_t>(x,y);
  };
  // 不好算范围，所以就不改变图像框大小了，（方案1）
  for(int i = 0;i < src.rows;i ++) {
    for(int j = 0;j < src.cols;j ++) {
      Eigen::Vector3d pose = Eigen::Vector3d::Zero();
      pose << i - src.rows / 2,j - src.cols / 2,1;
      dst.at<uint8_t>(i,j) = func(TT*pose);
    }
  }
}

int main(int argc, char **argv) {

  img_path = std::string(argv[1]);
  output = std::string(argv[2]);
  double angle = std::stod(argv[3]);
  
  cv::Mat img = cv::imread(img_path);
  cv::Mat gray;
  MY_IMG::Rbg2Gray(img,gray);
  std::cout << "gray img" << std::endl;

  cv::Mat dimg;
  rotate_img(gray,dimg,angle);

  cv::imshow("dimg",dimg);

  cv::waitKey(0);

  cv::imwrite(output,dimg);
  return 0;
}