#include "sift.h"
#include "unit.h"

namespace MY_IMG {
double _sigma(int s, double min_sigma, double max_sigma, double sigma_k) {
  // LOG("s : %d, sigma_k : %f", s, sigma_k);
  // LOG("--%lf", min_sigma * pow(2, s * sigma_k));
  return std::min(max_sigma, min_sigma * pow(2.0, s * sigma_k));
}
void _init_octave(Image &src, const SiftParam &param) {
  // -----------------------初始化第一层的图像--------------------------------
  LOG("-----------------------初始化第一层的图像-------------------------------"
      "-");
  IMG_Mat gray_img, init_image;
  if (src.img.channels() == 3) {
    Rgb2Gray(src.img, gray_img);
  } else {
    gray_img = src.img.clone();
  }

  IMG_Mat img_float = IMG_Mat(gray_img.size(), CV_32F);
  ConvertTo<uint8_t, float>(gray_img, img_float, 0,
                            1 / 255.0); // 归一化到[0,1]

  double sig_diff = 0;
  if (param.keep_appearance) {
    IMG_Mat tmp_img;
    ImageChange(img_float, tmp_img, 2.0, 0.0);
    sig_diff = sqrt(sqr(param.sigma) - 4.0 * sqr(param.init_sigma));
#ifdef USE_CONVOLUTION
    // 选择窗口大小（卷积滤波）
    int kernel_width =
        static_cast<int>(2 * param.gauss_kernel_patio * sig_diff) + 1;
    GaussBlur(tmp_img, init_image, sig_diff, kernel_width);
#endif
    tmp_img.release();
  } else {
    sig_diff = sqrt(sqr(param.sigma) - 1.0 * sqr(param.init_sigma));
#ifdef USE_CONVOLUTION
    // 选择窗口大小（卷积滤波）
    int kernel_width =
        static_cast<int>(2 * param.gauss_kernel_patio * sig_diff) + 1;
    GaussBlur(img_float, init_image, sig_diff, kernel_width);
#endif
  }
  gray_img.release();
  img_float.release();

  // -----------------------生成高斯金字塔--------------------------------
  LOG("-----------------------生成高斯金字塔--------------------------------");
  std::vector<double> sig;
  sig.push_back(param.sigma);
  double k = pow(2.0, 1.0 / param.num_octave_layers);
  for (int i = 1; i < param.num_octave_layers + 3; i++) {
    double prev_sig = param.sigma * pow(k, i - 1);
    double curr_sig = k * prev_sig;

    sig.push_back(sqrt(sqr(curr_sig) - sqr(prev_sig)));
    // sig.push_back(curr_sig);
  }
  for (auto &s : sig) {
    LOG("sigma : %lf", s);
  }
  src.Octaves.resize(param.num_octave);
  for (int i = 0; i < param.num_octave; i++) {
    src.Octaves[i].layers.resize(param.num_octave_layers + 3);
  }

  for (int i = 0; i < param.num_octave; i++) {
    for (int j = 0; j < param.num_octave_layers + 3; j++) {
      if (i == 0 && j == 0) {
        src.Octaves[i].layers[j] = init_image.clone();
      } else if (j == 0) {
        ImageChange(src.Octaves[i - 1].layers[3], src.Octaves[i].layers[0], 0.5,
                    0.0);
      } else {
#ifdef USE_CONVOLUTION
        // 选择窗口大小（卷积滤波）
        int kernel_width =
            static_cast<int>(2 * param.gauss_kernel_patio * sig[j - 1]) + 1;
        GaussBlur(src.Octaves[i].layers[j - 1], src.Octaves[i].layers[j],
                  sig[j], kernel_width);
#endif
      }
    }
  }
  init_image.release();
  sig.clear();
  // -----------------------生成高斯差分金字塔--------------------------------
  LOG("-----------------------生成高斯差分金字塔-------------------------------"
      "-");
  for (int i = 0; i < param.num_octave; i++) {
    for (int j = 0; j < param.num_octave_layers + 2; j++) {
      src.Octaves[i].dog_layers.push_back(src.Octaves[i].layers[j + 1] -
                                          src.Octaves[i].layers[j]);
    }
  }
}

bool _adjust_local_extrema(Image &src, const SiftParam &param, Point &kpt,
                           int o, int s) {
  float xi = 0, xr = 0, xc = 0;
  int i = 0;
  for (; i < param.max_interp_steps; i++) { // 迭代检测该点是否是最大值

    const IMG_Mat &img = src.Octaves[o].dog_layers[s]; // 当前层的图像
    const IMG_Mat &pre =
        src.Octaves[o].dog_layers[s - 1]; // 当前层的前一层的图像
    const IMG_Mat &nex =
        src.Octaves[o].dog_layers[s + 1]; // 当前层的后一层的图像

    // 获取特征点的一阶偏导
    float dx =
        (img.at<float>(kpt.x, kpt.y + 1) - img.at<float>(kpt.x, kpt.y - 1)) /
        2.0;
    float dy =
        (img.at<float>(kpt.x + 1, kpt.y) - img.at<float>(kpt.x - 1, kpt.y)) /
        2.0;
    float dz =
        (nex.at<float>(kpt.x, kpt.y) - pre.at<float>(kpt.x, kpt.y)) / 2.0;

    // 获取特征点的二阶偏导
    float v2 = img.at<float>(kpt.x, kpt.y);
    float dxx = (img.at<float>(kpt.x, kpt.y + 1) +
                 img.at<float>(kpt.x, kpt.y - 1) - 2 * v2);
    float dyy = (img.at<float>(kpt.x + 1, kpt.y) +
                 img.at<float>(kpt.x - 1, kpt.y) - 2 * v2);
    float dzz =
        pre.at<float>(kpt.x, kpt.y) + nex.at<float>(kpt.x, kpt.y) - 2 * v2;

    // 获取特征点二阶混合偏导
    float dxy = (img.at<float>(kpt.x + 1, kpt.y + 1) +
                 img.at<float>(kpt.x - 1, kpt.y - 1) -
                 img.at<float>(kpt.x + 1, kpt.y - 1) -
                 img.at<float>(kpt.x - 1, kpt.y + 1)) /
                4.0;
    float dxz =
        (nex.at<float>(kpt.x, kpt.y + 1) + pre.at<float>(kpt.x, kpt.y - 1) -
         nex.at<float>(kpt.x, kpt.y - 1) - pre.at<float>(kpt.x, kpt.y + 1)) /
        4.0;
    float dyz =
        (nex.at<float>(kpt.x + 1, kpt.y) + pre.at<float>(kpt.x - 1, kpt.y) -
         nex.at<float>(kpt.x - 1, kpt.y) - pre.at<float>(kpt.x + 1, kpt.y)) /
        4.0;

    Eigen::Matrix3d H;
    H << dxx, dxy, dxz, dxy, dyy, dyz, dxz, dyz, dzz;

    Eigen::Vector3d dD;
    dD << dx, dy, dz;

    Eigen::Vector3d X;
    // 利用SVD分解H矩阵得到特征点的最大值
    X = H.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(dD);

    xc = -X[0];
    xr = -X[1];
    xi = -X[2];

    // 如果三个方向偏移量都很小,则认为该点是准确的
    if (abs(xc) < 0.5 && abs(xr) < 0.5 && abs(xi) < 0.5) {
      break;
    }

    // 如果特征点的偏移量超过了最大偏移量,则认为该点是错误的
    if (abs(xc) > (float)(INT_MAX / 3) || abs(xr) > (float)(INT_MAX / 3) ||
        abs(xi) > (float)(INT_MAX / 3)) {
      return false;
    }

    // 开始偏移
    kpt.x += round(xc);
    kpt.y += round(xr);
    s += round(xi);

    // 如果偏移后的特征点超出了图像边界,则认为该点是错误的
    if (s < 1 || s > param.num_octave_layers || kpt.x < param.img_border ||
        kpt.x > img.rows - param.img_border || kpt.y < param.img_border ||
        kpt.y > img.cols - param.img_border) {
      return false;
    }
  }
  return true;
}

void _detect_keypoint(Image &src, const SiftParam &param) {
  // -----------------------检测极值点--------------------------------
  LOG("-----------------------检测极值点--------------------------------");
  float threshold = param.contrast_threshold / param.num_octave_layers;
  const int n = param.ori_hist_bins;
  float hist[n];
  std::vector<SiftPointDescriptor> kpt;

  src.keypoints.clear();
  int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1, 0};
  int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1, 0};

  for (int i = 0; i < param.num_octave; i++) {
    kpt.clear();
    for (int j = 1; j <= param.num_octave_layers; j++) {
      const IMG_Mat &curr_img = src.Octaves[i].layers[j];
      const IMG_Mat &prev_img = src.Octaves[i].layers[j - 1];
      const IMG_Mat &next_img = src.Octaves[i].layers[j + 1];
      LOG("curr_img[%ld %ld]",curr_img.rows,curr_img.cols);
      LOG("prev_img[%ld %ld]",prev_img.rows,prev_img.cols);
      LOG("next_img[%ld %ld]",next_img.rows,next_img.cols);
      int num_row = curr_img.rows;
      int num_col = curr_img.cols;
      bool min_x = true, max_x = true;

      for (int x = 0; x < num_row && (min_x || max_x); x++) {
        for (int y = 0; y < num_col && (min_x || max_x); y++) {
          float val = curr_img.at<float>(x, y);
          if (!(std::abs(val) > threshold)) {
            min_x = false;
            max_x = false;
            continue;
          }
          // 检测极值点
          for (int k = 0; k < 9; k++) {
            if ((val > prev_img.at<float>(x + dx[k], y + dy[k])) ||
                (val > next_img.at<float>(x + dx[k], y + dy[k])) ||
                (val > curr_img.at<float>(x + dx[k], y + dy[k]))) {
              min_x = false;
            }
            if ((val < prev_img.at<float>(x + dx[k], y + dy[k])) ||
                (val < next_img.at<float>(x + dx[k], y + dy[k])) ||
                (val < curr_img.at<float>(x + dx[k], y + dy[k]))) {
              max_x = false;
            }
          }
          Point pt(x, y);
          // 检测极值点
          // if (_adjust_local_extrema(src, param, pt, i, j)) {
          //   float zoom = pow(2.0, i - param.keep_appearance);
          //   kpt.push_back(SiftPointDescriptor(pt.x * zoom, pt.y * zoom));
          // }
            float zoom = pow(2.0, i - param.keep_appearance);
            kpt.push_back(SiftPointDescriptor(pt.x * zoom, pt.y * zoom,i,j));
        }
      }
    }
    LOG("detect point size : %d",kpt.size());
    for (auto p : kpt) {
      src.keypoints.push_back(p);
    }
  }
}

void SIFT(Image &src, std::vector<SiftPointDescriptor> &descriptors) {
  // -----------------------初始化参数--------------------------------
  LOG("初始化sift参数");
  SiftParam param;
  param.num_octave =
      std::min(static_cast<int>(log2(std::min(src.img.rows, src.img.cols))) - 3,
               param.max_octave);
  if (param.keep_appearance) {
    param.num_octave = std::min(param.num_octave + 1, param.max_octave);
  }
  param.num_octave_layers = 3;

  // -----------------------初始化图像金字塔--------------------------------
  LOG("初始化图像金字塔");
  _init_octave(src, param);

  // -----------------------检测关键点--------------------------------
  LOG("检测关键点");
  _detect_keypoint(src, param);

  // -----------------------计算描述子--------------------------------

  // -----------------------show--------------------------------
  // show
  LOG("keypoints size:%ld", src.keypoints.size());
  for (int i = 0; i < src.keypoints.size(); i++) {
    LOG("keypoint:[%d %d]", src.keypoints[i].keypoint.x,
        src.keypoints[i].keypoint.y);
  }
  int _o = 1;
  IMG_Mat tmp = IMG_Mat(src.Octaves[_o].layers[0].size(), CV_8U);
  for (int i = 0; i < param.num_octave_layers + 3; i++) {
    ConvertTo<float, uint8_t>(src.Octaves[_o].layers[i], tmp, 0, 255.0);
    cv::imshow("octave-" + std::to_string(_o) + "-layer-" + std::to_string(i),
               tmp);
  }
  for (int i = 0; i < param.num_octave_layers + 2; i++) {
    ConvertTo<float, uint8_t>(src.Octaves[_o].dog_layers[i], tmp, 0, 255.0);
    cv::imshow("octave-" + std::to_string(_o) + "-dog-" + std::to_string(i),
               tmp);
  }
  tmp.release();
  DrawPoints(src.img, src.keypoints, tmp);
  cv::imshow("keypoints", tmp);
  cv::waitKey(0);
}

} // namespace MY_IMG