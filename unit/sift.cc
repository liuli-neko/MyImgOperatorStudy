#include "sift.h"
#include "unit.h"

namespace MY_IMG {
double _sigma(int s, double min_sigma, double max_sigma, double sigma_k) {
  // LOG("s : %d, sigma_k : %f", s, sigma_k);
  // LOG("--%lf", min_sigma * pow(2, s * sigma_k));
  return std::min(max_sigma, min_sigma * pow(2.0, s * sigma_k));
}
void _init_octave(Image &src,const SiftParam &param) {
  // -----------------------初始化图像金字塔--------------------------------
  LOG("分配图像金字塔内存");
  // 初始化图像金字塔
  src.Octaves.resize(param.octave_num);
  
  // -----------------------初始化用于计算层内各图的滤波半径--------------------------------
  LOG("计算sigma");
  std::vector<double> sigmas;
  for (int j = 1; j < param.octave_layer_num; ++j) {
    sigmas.push_back(_sigma(j, param.octave_layer_min_sigma,
                            param.octave_layer_max_sigma,
                            param.octave_layer_sigma_k));
  }
  for (auto sigma : sigmas) {
    LOG("sigma : %lf", sigma);
  }

  // -----------------------初始化各层的图像集--------------------------------
  for (int i = 0; i < param.octave_num; ++i) {
    src.Octaves.at(i).layers.resize(1);
    // 计算金字塔每层的高斯模糊
    if (i == 0) { // 初始化第一层
      MY_IMG::Rgb2Gray(src.img, src.Octaves.at(i).layers.at(0),
                       {1 / 3.0, 1 / 3.0, 1 / 3.0});
    } else {
      MY_IMG::ImageChange(src.Octaves.at(i - 1).layers.at(0),
                          src.Octaves.at(i).layers.at(0), 1.0 / param.octave_k,
                          0.0);
    }
    // 计算金字塔每层的高斯模糊
    MY_IMG::ImageGaussianFilter(src.Octaves.at(i).layers.at(0),
                                src.Octaves.at(i).layers, sigmas);
    LOG("计算金字塔每层的DoG");
    src.Octaves.at(i).dog_layers.resize(param.octave_layer_num - 1);
    // 计算金字塔每层的DoG（高斯差分）
    for (int j = 0; j < param.octave_layer_num - 1; ++j) {
      src.Octaves.at(i).dog_layers.at(j).create(
          src.Octaves.at(i).layers.at(j).size(), CV_8UC1);
      for (int x = 0; x < src.Octaves.at(i).layers.at(j).rows; ++x) {
        for (int y = 0; y < src.Octaves.at(i).layers.at(j).cols; ++y) {
          float tmp_val =
              static_cast<float>(
                  src.Octaves.at(i).layers[j + 1].at<uint8_t>(x, y)) -
              src.Octaves.at(i).layers.at(j).at<uint8_t>(x, y);
          if (tmp_val < 0)
            tmp_val = 0;
          else if (tmp_val > 255)
            tmp_val = 255;
          src.Octaves.at(i).dog_layers.at(j).at<uint8_t>(x, y) = tmp_val;
        }
      }
      LOG("layers size : %d, %d", src.Octaves.at(i).layers.at(j).rows,
          src.Octaves.at(i).layers.at(j).cols);
    }
  }
}
void SIFT(Image &src, std::vector<SiftPointDescriptor> &descriptors) {
  // -----------------------初始化参数--------------------------------
  LOG("初始化sift参数");
  SiftParam param;
  param.octave_num =
      static_cast<int>(log2(std::min(src.img.rows, src.img.cols))) - 5;
  param.octave_layer_num = 4;
  param.octave_layer_min_sigma = 1.6;
  param.octave_layer_max_sigma = 3.2;
  param.octave_layer_sigma_k = 1.0 / param.octave_layer_num;

  // -----------------------初始化图像金字塔--------------------------------
  LOG("初始化图像金字塔");
  _init_octave(src, param);

  // show
  int _i = 0;
  cv::imshow("src0", src.Octaves.at(_i).layers.at(0));
  for (int j = 1; j < param.octave_layer_num; ++j) {
    cv::imshow("layer" + std::to_string(j), src.Octaves.at(_i).layers.at(j));
    cv::imshow("dog" + std::to_string(j - 1),
               src.Octaves.at(_i).dog_layers.at(j - 1));
  }
  cv::waitKey(0);
}

} // namespace MY_IMG