#ifndef IMG_OPERATOR_UNIT_CAMERA_H_
#define IMG_OPERATOR_UNIT_CAMERA_H_
#define USE_EIGEN
#include "all.h"
namespace MY_IMG {
class Camera {
public:
  Camera();

  Eigen::Vector2d ProjectPoint2Img(const Eigen::Vector3d &point_3dm,
                                   const bool point_in_world = true);
  Eigen::Matrix4d PoseWorld2Camera();

  void SetR(const Eigen::Matrix3d &R);
  void SetT(const Eigen::Vector3d &t);
  void SetCenter(const Eigen::Vector2d &center);
  void SetDistortion(const Eigen::VectorXd dist_offset);
  void SetFocalDistance(const std::vector<double> &focal_distance);

  Eigen::Matrix3d GetR();
  Eigen::Vector3d GetT();
  Eigen::Vector2d GetCenter();
  Eigen::VectorXd Getdistortion();
  std::vector<double> GetFocalDistance();

private:
  Eigen::Matrix3d R_;
  Eigen::Vector3d t_;
  Eigen::Vector2d center_;
  Eigen::VectorXd dist_offset_;
  std::vector<double> f_;
};
} // namespace MY_IMG
#endif