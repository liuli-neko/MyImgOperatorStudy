#include "sift.h"
#include "unit.h"

namespace MY_IMG {
double _sigma(int s, double min_sigma, double max_sigma, double sigma_k) {
  // LOG("s : %d, sigma_k : %f", s, sigma_k);
  // LOG("--%lf", min_sigma * pow(2, s * sigma_k));
  return std::min(max_sigma, min_sigma * pow(2.0, s * sigma_k));
}
void _init_octave(Image &src, const SiftParam &param) {
  // -----------------------初始化图像金字塔--------------------------------
  LOG("分配图像金字塔内存");
  // 初始化图像金字塔
  src.Octaves.resize(param.num_octave);

  // -----------------------初始化第一层的图像--------------------------------
  IMG_Mat gray_img;
  if (src.img.channels() == 3) {
    MY_IMG::Rgb2Gray(src.img, gray_img);
  } else {
    gray_img = src.img.clone();
  }

  


}

void _detect_keypoint(Image &src, const SiftParam &param) {
  // -----------------------检测极值点--------------------------------

}

void SIFT(Image &src, std::vector<SiftPointDescriptor> &descriptors) {
  // -----------------------初始化参数--------------------------------
  LOG("初始化sift参数");
  SiftParam param;
  param.num_octave =
      std::min(static_cast<int>(log2(std::min(src.img.rows, src.img.cols))) - 3,
               param.max_octave);
  if(param.keep_appearance) {
    param.num_octave = std::min(param.num_octave + 1, param.max_octave);
  }

  // -----------------------初始化图像金字塔--------------------------------
  LOG("初始化图像金字塔");
  _init_octave(src, param);

  // -----------------------检测关键点--------------------------------
  LOG("检测关键点");
  _detect_keypoint(src, param);

  // -----------------------计算描述子--------------------------------

  // -----------------------show--------------------------------
  // show

}

} // namespace MY_IMG