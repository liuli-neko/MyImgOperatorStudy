#ifndef IMG_OPERATOR_UNIT_SIFT_H_
#define IMG_OPERATOR_UNIT_SIFT_H_

#include "image_operator.h"
#include "all.h"
#define USE_FILTER 1

namespace MY_IMG {

// 定义sift的参数结构体
struct SiftParam {
  int max_features = 0; // 最大特征点数量,0表示不限制
  int num_octave_layers = 3; // 图像金字塔内每一层的层数
  double contrast_threshold = 0.03; // 对比度阈值(D(x))
  double edge_threshold = 10; // 边缘阈值(E(x))
  double sigma = 1.6; // 卷积核的标准差
  bool keep_appearance = true; // 是否保留原始图像
  double gauss_kernel_patio = 3; // 高斯核的尺寸size=2*gauss_kernel_patio*simga+1
  int max_octave = 10; // 最大octave数
  int num_octave = 5; // 图像金字塔的层数
  float contr_threshold = 0.03; // 关键点的阈值
  float curv_threshold = 10.0; // 关键点的阈值
  float init_sigma = 0.5; // 初始sigma
  int img_border = 2; // 图像边界忽略宽度
  int max_interp_steps = 5; // 关键点精确插值字数
  int ori_hist_bins = 36; // 关键点方向直方图bin数
  float ori_sig_fctr = 1.5; // 关键点主方向直方图的sigma参数
  float ori_radius = 3 * ori_sig_fctr; // 关键点主方向直方图的半径
  float ori_peak_ratio = 0.8; // 关键点主方向直方图的高度阈值
  int descr_width = 4; // 描述子网格宽度
  int descr_hist_bins = 8; // 描述子直方图方向的维度
  float descr_mag_thr = 0.2; // 描述子幅度阈值
  float descr_scl_fctr = 3.0; // 描述子网格大小
  int sift_fixpt_scale = 1; 
};

/** @brief 特征提取和描述子生成
 * @param img 图像数据
 * @param param sift参数
 * @param descriptors 提取出的关键点与其描述子
*/
void SIFT(Image &img);
void FeatureExtraction(Image &src, const SiftParam &param);
} // namespace MY_IMG

#endif