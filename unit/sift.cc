#include "sift.h"
#include "unit.h"

namespace MY_IMG {
void _init_octave_filter(Image &src, const SiftParam &param) {
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

  ConvertTo<uint8_t, float>(gray_img, img_float,
                            [](const uint8_t &pixe) -> float {
                              return static_cast<float>(pixe) / 255.0f;
                            }); // 归一化到(0,1)

  double sig_diff = 0;
  if (param.keep_appearance) {
    IMG_Mat tmp_img;
    ImageChange(img_float, tmp_img, 2.0, 0.0);
    sig_diff = sqrt(sqr(param.sigma) - 4.0 * sqr(param.init_sigma));
    // 选择窗口大小（卷积滤波）
    int kernel_width =
        static_cast<int>(2 * param.gauss_kernel_patio * sig_diff) + 1;
    GaussBlur(tmp_img, init_image, sig_diff, kernel_width);
    tmp_img.release();
  } else {
    sig_diff = sqrt(sqr(param.sigma) - 1.0 * sqr(param.init_sigma));
    // 选择窗口大小（卷积滤波）
    int kernel_width =
        static_cast<int>(2 * param.gauss_kernel_patio * sig_diff) + 1;
    GaussBlur(img_float, init_image, sig_diff, kernel_width);
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
        // 选择窗口大小（卷积滤波）
        int kernel_width =
            static_cast<int>(2 * param.gauss_kernel_patio * sig[j - 1]) + 1;
        GaussBlur(src.Octaves[i].layers[j - 1], src.Octaves[i].layers[j],
                  sig[j], kernel_width);
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

void _init_octave_fft(Image &src, const SiftParam &param) {
  // -----------------------初始化第一层的图像--------------------------------
  LOG("-----------------------初始化第一层的图像-------------------------------"
      "-");
  IMG_Mat gray_img, init_image;
  if (src.img.channels() == 3) {
    Rgb2Gray(src.img, gray_img);
  } else {
    gray_img = src.img.clone();
  }

  double sig_diff = 0;
  if (param.keep_appearance) {
    IMG_Mat tmp_img;
    ImageChange(gray_img, tmp_img, 2.0, 0.0);
    sig_diff = sqrt(sqr(param.sigma) - 4.0 * sqr(param.init_sigma));
    ImageGaussianFilter(tmp_img, init_image, sig_diff);
    tmp_img.release();
  } else {
    sig_diff = sqrt(sqr(param.sigma) - 1.0 * sqr(param.init_sigma));
    ImageGaussianFilter(gray_img, init_image, sig_diff);
  }
  gray_img.release();

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
        ImageGaussianFilter(ConvertSingleChannelMat2Uint8Mat<float>(
                                src.Octaves[i].layers[j - 1]),
                            src.Octaves[i].layers[j], sig[j]);
      }
    }
  }
  init_image.release();
  sig.clear();
  // -----------------------生成高斯差分金字塔--------------------------------
  LOG("-----------------------生成高斯差分金字塔-------------------------------"
      "-");
  for (int i = 0; i < param.num_octave; i++) {
    for (int j = 0; j < param.num_octave_layers + 3; j++) {
      src.Octaves[i].layers[j] /= 255.0;
    }
    for (int j = 0; j < param.num_octave_layers + 2; j++) {
      src.Octaves[i].dog_layers.push_back(src.Octaves[i].layers[j + 1] -
                                          src.Octaves[i].layers[j]);
    }
  }
}

bool _adjust_local_extrema(Image &src, const SiftParam &param, KeyPoint &kpt,
                           int octave, int &layer, int &row, int &col) {
  // -----------------------迭代更新关键点位置-------------------------------
  const float img_scale = 1.0f / (255 * param.sift_fixpt_scale);
  const float deriv_scale = img_scale * 0.5f;
  const float second_deriv_scale = img_scale;
  const float cross_deriv_scale = img_scale * 0.25f;

  float xi = 0, xr = 0, xc = 0;
  int i = 0;

  for (; i < param.max_interp_steps; i++) { // 迭代检测该点是否是最大值

    const IMG_Mat &img = src.Octaves[octave].dog_layers[layer]; // 当前层的图像
    const IMG_Mat &pre =
        src.Octaves[octave].dog_layers[layer - 1]; // 当前层的前一层的图像
    const IMG_Mat &nex =
        src.Octaves[octave].dog_layers[layer + 1]; // 当前层的后一层的图像

    // 获取特征点的一阶偏导
    float dx =
        (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) / 2.0;
    float dy =
        (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) / 2.0;
    float dz = (nex.at<float>(row, col) - pre.at<float>(row, col)) / 2.0;

    // 获取特征点的二阶偏导
    float v2 = img.at<float>(row, col);
    float dxx =
        (img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - 2 * v2);
    float dyy =
        (img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - 2 * v2);
    float dzz = pre.at<float>(row, col) + nex.at<float>(row, col) - 2 * v2;

    // 获取特征点二阶混合偏导
    float dxy =
        (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
         img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) /
        4.0;
    float dxz = (nex.at<float>(row, col + 1) + pre.at<float>(row, col - 1) -
                 nex.at<float>(row, col - 1) - pre.at<float>(row, col + 1)) /
                4.0;
    float dyz = (nex.at<float>(row + 1, col) + pre.at<float>(row - 1, col) -
                 nex.at<float>(row - 1, col) - pre.at<float>(row + 1, col)) /
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
    col += round(xc);
    row += round(xr);
    layer += round(xi);

    // 如果偏移后的特征点超出了图像边界,则认为该点是错误的
    if (layer < 1 || layer > param.num_octave_layers ||
        col < param.img_border || col > img.rows - param.img_border ||
        row < param.img_border || row > img.cols - param.img_border) {
      return false;
    }
  }
  // 如果最后也没找到合适的点,则认为该点是错误的
  if (i >= param.max_interp_steps - 1) {
    return false;
  }

  // ------------------------删除不那么突出的点----------------------
  // 计算一阶导数
  const IMG_Mat &img = src.Octaves[octave].dog_layers[layer];
  const IMG_Mat &pre = src.Octaves[octave].dog_layers[layer - 1];
  const IMG_Mat &nex = src.Octaves[octave].dog_layers[layer + 1];

  float dx =
      (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) * deriv_scale;
  float dy =
      (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) * deriv_scale;
  float dz = (nex.at<float>(row, col) - pre.at<float>(row, col)) * deriv_scale;

  Eigen::Vector3f dD;
  dD << dx, dy, dz;

  float t = dD.dot(Eigen::Vector3f(xc, xr, xi));

  float contr = img.at<float>(row, col) + t * 0.5f;

  if (std::abs(contr) < param.contrast_threshold / param.num_octave_layers) {
    return false;
  }

  // -----------------------删除边缘相应比较强的点-----------------------
  // 计算Hessian 矩阵
  float v2 = img.at<float>(row, col) * 2.0f;
  float dxx = (img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - v2) *
              second_deriv_scale;
  float dyy = (img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - v2) *
              second_deriv_scale;
  float dxy =
      (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
       img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) *
      second_deriv_scale;
  float det = dxx * dyy - dxy * dxy;
  float trace = dxx + dyy;

  // 主曲率和阈值
  if (det <= 0 ||
      trace * trace * param.edge_threshold >=
          det * (param.edge_threshold + 1) * (param.edge_threshold + 1)) {
    return false;
  }

  // ------------------------------保存特征点-----------------------------------
  kpt.x = ((float)row + xc) * pow(2, octave - param.keep_appearance);
  kpt.y = ((float)col + xr) * pow(2, octave - param.keep_appearance);

  // SIFT 描述子
  kpt.octave = octave + (layer << 8) + (int(xi + 0.5) << 16);
  kpt.size = param.sigma * powf(2.0f, (layer + xi) / param.num_octave_layers) *
             (1 << octave) * 2;
  kpt.response = abs(contr);

  return true;
}

float _calc_orientation_hist(const IMG_Mat &img, const SiftParam &param,
                             KeyPoint &pt, float scale, int n) {
  // -----------------------------计算描述子--------------------------------
  std::vector<float> &hist = pt.descriptor;
  hist.resize(n); // 初始化描述子
  int radius = (int)(param.ori_radius * scale + 0.5);

  int len = sqr(2 * radius + 1);

  float sigma = param.ori_sig_fctr * scale;

  float exp_scale = -1.0f / (2 * sigma * sigma);

  float *buffer = new float[4 * len + n + 4];
  float *X = buffer, *Y = buffer + len, *Mag = Y, *Ori = Y + len,
        *W = Ori + len;
  float *temp_hist = W + len + 2;

  for (int i = 0; i < n; i++) {
    temp_hist[i] = 0.0f;
  }

  int k = 0;

  for (int i = -radius; i <= radius; i++) {
    int y = pt.y + i;

    if (y <= 0 || y >= img.rows - 1) {
      continue;
    }
    for (int j = -radius; j <= radius; j++) {
      int x = pt.x + j;

      if (x <= 0 || x >= img.cols - 1) {
        continue;
      }

      float dx = img.at<float>(y, x + 1) - img.at<float>(y, x - 1);
      float dy = img.at<float>(y + 1, x) - img.at<float>(y - 1, x);

      X[k] = dx;
      Y[k] = dy;
      W[k] = (sqr(i) + sqr(j)) * exp_scale;

      k++;
    }
  }

  len = k;

  for (int i = 0; i < len; i++) {
    // 计算领域内的所有像素的高斯权重
    W[i] = std::exp(W[i]);
  }

  for (int i = 0; i < len; i++) {
    // 计算领域内所有像素的梯度方向
    Ori[i] = std::atan2(Y[i], X[i]);
  }

  for (int i = 0; i < len; i++) {
    // 计算领域内的所有像素的幅度
    Mag[i] = std::sqrt(sqr(X[i]) + sqr(Y[i]));
  }

  // 遍历领域的像素
  for (int i = 0; i < len; i++) {
    int bin = (int)((n / 360.0f) * Ori[i] + 0.5f);

    if (bin >= n) {
      bin = bin - n;
    }

    if (bin < 0) {
      bin = bin + n;
    }

    temp_hist[bin] = temp_hist[bin] + Mag[i] * W[i];
  }

  // 平滑直方图
  temp_hist[-1] = temp_hist[n - 1];
  temp_hist[-2] = temp_hist[n - 2];
  temp_hist[n] = temp_hist[0];
  temp_hist[n + 1] = temp_hist[1];
  for (int i = 0; i < n; i++) {
    hist[i] = (temp_hist[i - 2] + temp_hist[i + 2]) * (1.0f / 16.0f) +
              (temp_hist[i - 1] + temp_hist[i + 1]) * (4.0f / 16.0f) +
              temp_hist[i] * (6.0f / 16.0f);
  }

  // 获取直方图中的最大值
  float max_val = hist[0];
  for (int i = 1; i < n; i++) {
    if (hist[i] > max_val) {
      max_val = hist[i];
    }
  }
  delete[] buffer;
  return max_val;
}

void _detect_keypoint(Image &src, const SiftParam &param) {
  // -----------------------检测极值点--------------------------------
  LOG("-----------------------检测极值点--------------------------------");
  float threshold = param.contrast_threshold / param.num_octave_layers;
  const int n = param.ori_hist_bins;
  float hist[n];
  std::vector<KeyPoint> kpt;

  src.keypoints.clear();
  int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1, 0};
  int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1, 0};

  for (int i = 0; i < param.num_octave; i++) {
    kpt.clear();
    int numKeys = 0;
    for (int j = 1; j <= param.num_octave_layers; j++) {
      const IMG_Mat &curr_img = src.Octaves[i].dog_layers[j];
      const IMG_Mat &prev_img = src.Octaves[i].dog_layers[j - 1];
      const IMG_Mat &next_img = src.Octaves[i].dog_layers[j + 1];

      int num_row = curr_img.rows;
      int num_col = curr_img.cols;

      for (int x = 0; x < num_row; x++) {
        for (int y = 0; y < num_col; y++) {
          bool min_x = true, max_x = true;
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
          if (min_x || max_x) {
            numKeys++;
            // 检测极值点
            KeyPoint kp;
            int octave = i, layer = j, r1 = x, c1 = y;
            if (_adjust_local_extrema(src, param, kp, octave, layer, r1, c1)) {
              // ASSERT(kp.x == r1 && kp.y == c1,"kp(%d %d) != (%d %d)", kp.x,
              // kp.y, r1, c1);

              float scale = kp.size / float(1 << octave);

              // 计算关键点的主方向
              float max_hist = _calc_orientation_hist(
                  src.Octaves[octave].layers[layer], param, kp, scale, n);

              float sum = 0.0;
              float mag_thr = 0.0;

              for (int i = 0; i < n; i++) {
                sum += kp.descriptor[i];
              }
              mag_thr = 0.5 * (1.0 / 36) * sum;

              // 遍历所有bin
              for (int i = 0; i < n; i++) {
                int left = i > 0 ? i - 1 : n - 1;
                int right = i < n - 1 ? i + 1 : 0;

                //创建新的特征点，大于主峰的80%
                if (kp.descriptor[i] > kp.descriptor[left] &&
                    kp.descriptor[i] > kp.descriptor[right] &&
                    kp.descriptor[i] >= mag_thr) {
                  float bin =
                      i + 0.5f * (kp.descriptor[left] - kp.descriptor[right]) /
                              (kp.descriptor[left] + kp.descriptor[right] -
                               2 * kp.descriptor[i]);
                  bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;

                  // 对主方向做一个细化
                  float angle = (360.0f / n) * bin;
                  if (angle >= 1 && angle <= 180) {
                    kp.angle = angle;
                  } else if (angle > 180 && angle <= 360) {
                    kp.angle = 360 - angle;
                  }
                  kpt.push_back(kp);
                }
              }
            }
          }
        }
      }
    }
    LOG("numKeys = %d", numKeys);
    LOG("detect point size : %ld", kpt.size());
    for (auto p : kpt) {
      src.keypoints.push_back(p);
    }
    kpt.clear();
  }
}

void SIFT(Image &src, std::vector<KeyPoint> &descriptors) {
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
#if USE_FILTER == 1
  _init_octave_filter(src, param);
#elif USE_FILTER == 2
  _init_octave_fft(src, param);
#endif

  // -----------------------获取关键点--------------------------------
  LOG("检测关键点");
  _detect_keypoint(src, param);
  // 如果设置了特征点个数限制，则进行剪裁
  if (param.max_features != 0 && src.keypoints.size() > param.max_features) {
    std::sort(src.keypoints.begin(), src.keypoints.end(),
              [](const KeyPoint &a, const KeyPoint &b) {
                return a.response > b.response;
              });
    // 删除多余的特征点
    src.keypoints.erase(src.keypoints.begin() + param.max_features,
                        src.keypoints.end());
  }
  // -----------------------show--------------------------------
  // show
#ifdef DEBUG
  LOG("keypoints size:%ld", src.keypoints.size());
  int _o = 1;
  IMG_Mat tmp = IMG_Mat(src.Octaves[_o].layers[0].size(), CV_8U);
  for (int i = 0; i < param.num_octave_layers + 3; i++) {
    ConvertTo<float, uint8_t>(src.Octaves[_o].layers[i], tmp,
                              [](const float &pixe) -> uint8_t {
                                return static_cast<uint8_t>(pixe * 255);
                              });
    cv::imshow("octave-" + std::to_string(_o) + "-layer-" + std::to_string(i),
               tmp);
  }
  for (int i = 0; i < param.num_octave_layers + 2; i++) {
    ConvertTo<float, uint8_t>(
        src.Octaves[_o].dog_layers[i], tmp,
        [](const float &pixe) -> uint8_t { return pixe * 255; });
    cv::imshow("octave-" + std::to_string(_o) + "-dog-" + std::to_string(i),
               tmp);
  }
  tmp.release();

  DrawPoints(src.img, src.keypoints, tmp);
  cv::imshow("keypoints", tmp);
  cv::waitKey(0);
#endif
  // -----------------------释放图像金字塔内存--------------------------------
  LOG("释放图像金字塔内存");
  src.Octaves.clear();
}

} // namespace MY_IMG