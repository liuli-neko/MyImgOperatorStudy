#include "camera.h"
#include <iostream>
namespace MY_IMG {
Camera::Camera() {
  R_ = Eigen::Matrix3d::Identity();
  t_ = Eigen::Vector3d::Zero();
  center_ = Eigen::Vector2d::Zero();
  f_ = std::vector<double>(2, 1.0);
}
Eigen::Vector2d Camera::ProjectPoint2Img(const Eigen::Vector3d &point_3d,
                                         const bool point_in_world) {
  LOG(INFO,"point_3d : (%.2lf,%.2lf,%.2lf)", point_3d[0], point_3d[1], point_3d[2]);
  Eigen::Vector3d camera_point = point_3d;
  if (point_in_world) {
    // 将点转到camear坐标系
    camera_point = R_ * camera_point + t_;
  }
  // 将点投影到归1化平面
  Eigen::Vector2d point;
  point << camera_point[0] / camera_point[2], camera_point[1] / camera_point[2];

  // 计算畸变过程
  double r2 = point[0] * point[0] + point[1] * point[1], rx = 1;
  double f = 1;
  for (double i = 0; i < dist_offset_.rows(); i++) {
    rx *= r2;
    f += rx * dist_offset_[i];
  }
  // 投影到像平面
  if (f_.size() == 2) {
    point[0] *= f_[0];
    point[1] *= f_[1];
  } else if (f_.size() == 1) {
    point[0] *= f_[0];
    point[1] *= f_[0];
  }

  // 计算畸变过程
  point *= f;

  // 加上相机中心的偏移
  point += center_;
  LOG(INFO,"point : (%.2lf,%.2lf)", point[0], point[1]);
  return point;
}
Eigen::Matrix4d Camera::PoseWorld2Camera() {
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

  pose.block<3, 3>(0, 0) = R_;
  pose.block<3, 1>(0, 3) = t_;

  return pose;
}
void Camera::SetR(const Eigen::Matrix3d &R) { R_ = R; }
void Camera::SetT(const Eigen::Vector3d &t) { t_ = t; }
void Camera::SetCenter(const Eigen::Vector2d &center) { center_ = center; }
void Camera::SetDistortion(const Eigen::VectorXd dist_offset) {
  dist_offset_ = dist_offset;
}
void Camera::SetFocalDistance(const std::vector<double> &f) { f_ = f; }
Eigen::Matrix3d Camera::GetR() { return R_; }
Eigen::Vector3d Camera::GetT() { return t_; }
Eigen::Vector2d Camera::GetCenter() { return center_; }
Eigen::VectorXd Camera::Getdistortion() { return dist_offset_; }
std::vector<double> Camera::GetFocalDistance() { return f_; }
} // namespace MY_IMG