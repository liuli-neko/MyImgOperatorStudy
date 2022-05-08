#ifndef IMG_OPERATOR_UNIT_SIFT_H_
#define IMG_OPERATOR_UNIT_SIFT_H_

#include "image_operator.h"
#include "all.h"

namespace MY_IMG {
// 用与保存图像金字塔的结构体
struct Octave {
  std::vector<IMG_Mat> layers;
  std::vector<IMG_Mat> dog_layers;
  void release() {
    for (int i = 0; i < layers.size(); ++i) {
      layers[i].release();
    }
    for (int i = 0; i < dog_layers.size(); ++i) {
      dog_layers[i].release();
    }
  }
};
// 定义用于保存原始图片的结构体
struct Image {
  IMG_Mat img;
  int imgId;
  std::vector<Octave> Octaves; // octave*layer
  std::vector<Point2d> keypoints;
};
// 定义用于存储sift描述子的结构体
struct SiftPointDescriptor
{
  int16_t imgId;
  Point2d keypoint;
  double descriptor[128];
};
// 定义sift的参数结构体
struct SiftParam {
  int octave_num = 4;
  int octave_layer_num = 5;
  double octave_layer_min_sigma = 1.6;
  double octave_layer_max_sigma = 20;
  double octave_layer_sigma_k = 1.0/5;
  double octave_layer_sigma_step = 0;
  double octave_k = 2.0;
  double Hessian_r = 10;
};
/** @brief 特征提取和描述子生成
 * @param img 图像数据
 * @param param sift参数
 * @param descriptors 提取出的关键点与其描述子
*/
void SIFT(Image &img,std::vector<SiftPointDescriptor> &descriptors);

} // namespace MY_IMG

#endif