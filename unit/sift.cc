#include "sift.h"
#include "all.h"
#include "unit.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

namespace MY_IMG {
void _init_sigmas(const SiftParam &param, std::vector<double> &sigmas) {
  sigmas.clear();
  sigmas.push_back(param.sigma);
  double k = pow(2.0, 1.0 / param.num_octave_layers);
  for (int i = 1; i < param.num_octave_layers + 3; i++) {
    double prev_sig = param.sigma * pow(k, i - 1);
    double curr_sig = k * prev_sig;

    sigmas.push_back(sqrt(sqr(curr_sig) - sqr(prev_sig)));
    // sig.push_back(curr_sig);
  }
  for (auto &s : sigmas) {
    LOG(INFO, "sigma : %lf", s);
  }
}
void _init_first_img(const Image &src, const SiftParam &param, IMG_Mat &img) {
  img.create(src.img.rows, src.img.cols, CV_32F);
  if (src.img.channels() == 3) {
    ConvertTo<cv::Vec3b, float>(src.img, img, [](const cv::Vec3b &v) -> float {
      return (float)v[0] * 0.299 + (float)v[1] * 0.587 + (float)v[2] * 0.114;
    });
  } else {
    ConvertTo<uchar, float>(src.img, img,
                            [](const uchar &v) -> float { return (float)v; });
  }
  double sig_diff = 0;
  if (param.keep_appearance) {
    ImageChange(img, img, 2.0, 0);
    sig_diff = sqrt(sqr(param.sigma) - 4.0 * sqr(param.init_sigma));
    if (USE_FILTER == 1) {
      int kernel_width =
          static_cast<int>(2 * param.gauss_kernel_patio * sig_diff) + 1;
      GaussBlur(img, img, sig_diff, kernel_width);
    } else if (USE_FILTER == 2) {
      ImageGaussianFilter(ConvertSingleChannelMat2Uint8Mat<float>(img), img,
                          sig_diff);
    }
  } else {
    sig_diff = sqrt(sqr(param.sigma) - 1.0 * sqr(param.init_sigma));
    if (USE_FILTER == 1) {
      int kernel_width =
          static_cast<int>(2 * param.gauss_kernel_patio * sig_diff) + 1;
      GaussBlur(img, img, sig_diff, kernel_width);
    } else if (USE_FILTER == 2) {
      ImageGaussianFilter(ConvertSingleChannelMat2Uint8Mat<float>(img), img,
                          sig_diff);
    }
  }
}
void _init_octave_gauss_pyramid(Image &src, const SiftParam &param,
                                const std::vector<double> &sigmas,
                                const IMG_Mat &first_img,
                                std::vector<Octave> &Octaves) {
  int num_octaves = param.num_octave;
  int num_octave_layers = param.num_octave_layers + 3;
  Octaves.resize(num_octaves);
  for (int i = 0; i < num_octaves;i ++ ) {
    Octaves[i].layers.resize(num_octave_layers);
  }

  for (int i = 0; i < num_octaves; i++) {
    for (int j = 0; j < num_octave_layers; j++) {
      if (i == 0 && j == 0) {
        Octaves[i].layers[j] = first_img.clone();
      } else if (j == 0) {
        ImageChange(Octaves[i - 1].layers[3], Octaves[i].layers[0], 0.5, 0.0);
      } else {
        if (USE_FILTER == 1) {
          int kernel_width =
              static_cast<int>(2 * param.gauss_kernel_patio * sigmas[j]) + 1;
          GaussBlur(Octaves[i].layers[j - 1], Octaves[i].layers[j], sigmas[j],
                    kernel_width);
        } else if (USE_FILTER == 2) {
          ImageGaussianFilter(
              ConvertSingleChannelMat2Uint8Mat<float>(Octaves[i].layers[j - 1]),
              Octaves[i].layers[j], sigmas[j]);
        }
      }
    }
  }
  // 归一化
  for (int i = 0; i < num_octaves; i++) {
    for (int j = 0; j < num_octave_layers; j++) {
      Octaves[i].layers[j] /= 255.0;
    }
  }
}
void _init_dog_pyramid(Image &src, std::vector<Octave> &Octaves,
                       const SiftParam &param) {
  int num_octaves = param.num_octave;
  int num_octave_layers = param.num_octave_layers + 2;
  for (int i = 0; i < num_octaves; i++) {
    for (int j = 0; j < num_octave_layers; j++) {
      Octaves[i].dog_layers[j] =
          Octaves[i].layers[j + 1] - Octaves[i].layers[j];
    }
  }
}
bool _adjust_local_extrema(Image &src, const std::vector<Octave> &Octaves,
                           const SiftParam &param, std::shared_ptr<KeyPoint> &kp,
                           int octave, int &layer, int &row, int &col) {
  // -----------------------迭代更新关键点位置-------------------------------
  const float img_scale = 1.0f / (255 * param.sift_fixpt_scale);
  const float deriv_scale = img_scale * 0.5f;
  const float second_deriv_scale = img_scale;
  const float cross_deriv_scale = img_scale * 0.25f;

  float xi = 0, xr = 0, xc = 0;
  int i = 0;

  for (; i < param.max_interp_steps; i++) { // 迭代检测该点是否是最大值

    const IMG_Mat &img = Octaves[octave].dog_layers[layer]; // 当前层的图像
    const IMG_Mat &pre =
        Octaves[octave].dog_layers[layer - 1]; // 当前层的前一层的图像
    const IMG_Mat &nex =
        Octaves[octave].dog_layers[layer + 1]; // 当前层的后一层的图像
    ASSERT(row > 0 && row < img.rows - 1 && col > 0 && col < img.cols - 1,
           "row: %d, col: %d", row, col);
    // 获取特征点的一阶偏导
    float dx =
        (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) * deriv_scale;
    float dy =
        (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) * deriv_scale;
    float dz = (nex.at<float>(row, col) - pre.at<float>(row, col)) * deriv_scale;

    // 获取特征点的二阶偏导
    float v2 = img.at<float>(row, col) * 2;
    float dxx =
        (img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - v2) * second_deriv_scale;
    float dyy =
        (img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - v2) * second_deriv_scale;
    float dzz = (pre.at<float>(row, col) + nex.at<float>(row, col) - v2) * second_deriv_scale;

    // 获取特征点二阶混合偏导
    float dxy =
                (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
                img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) *
                second_deriv_scale;
    float dxz = (nex.at<float>(row, col + 1) + pre.at<float>(row, col - 1) -
                 nex.at<float>(row, col - 1) - pre.at<float>(row, col + 1)) *
                 second_deriv_scale;
    float dyz = (nex.at<float>(row + 1, col) + pre.at<float>(row - 1, col) -
                 nex.at<float>(row - 1, col) - pre.at<float>(row + 1, col)) *
                 second_deriv_scale;

/* 不是求特征值？
    Eigen::Matrix3d H;
    H << dxx, dxy, dxz, dxy, dyy, dyz, dxz, dyz, dzz;

    Eigen::Vector3d dD;
    dD << dx, dy, dz;

    Eigen::Vector3d X;
    // 利用SVD分解H矩阵得到特征点的最大值
    X = H.colPivHouseholderQr().solve(dD);
*/
    cv::Vec3f dD(dx, dy, dz);
    cv::Matx33f H(dxx, dxy, dxz, dxy, dyy, dyz, dxz, dyz, dzz);

    cv::Vec3f X = H.solve(dD, cv::DECOMP_LU);

    xc = -X[0];
    xr = -X[1];
    xi = -X[2];

    // 如果三个方向偏移量都很小,则认为该点是准确的
    if (abs(xc) < 0.5f && abs(xr) < 0.5f && abs(xi) < 0.5f) {
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
        col < param.img_border || col > img.cols - param.img_border ||
        row < param.img_border || row > img.rows - param.img_border) {
      return false;
    }
  }
  // 如果最后也没找到合适的点,则认为该点是错误的
  if (i >= param.max_interp_steps) {
    return false;
  }

  // ------------------------删除不那么突出的点----------------------
  // 计算一阶导数
  const IMG_Mat &img = Octaves[octave].dog_layers[layer];
  const IMG_Mat &pre = Octaves[octave].dog_layers[layer - 1];
  const IMG_Mat &nex = Octaves[octave].dog_layers[layer + 1];

  float dx =
      (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) * deriv_scale;
  float dy =
      (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) * deriv_scale;
  float dz = (nex.at<float>(row, col) - pre.at<float>(row, col)) * deriv_scale;

  cv::Matx31f dD(dx, dy, dz);

  float t = dD.dot(cv::Matx31f(xc, xr, xi));

  float contr = img.at<float>(row, col) + t * 0.5f;

  if (std::abs(contr) * param.num_octave_layers < param.contrast_threshold) {
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
  // LOG(DEBUG,"x : %f, y : %f, layer : %d", col + xc, row + xr, layer);
  kp->x = ((float)row + xc) * (1 << octave);
  kp->y = ((float)col + xr) * (1 << octave);
  // LOG(DEBUG,"kp(%d, %d)", kp->x, kp->y);

  // SIFT 描述子
  kp->octave = octave + (layer << 8) + (int((xi + 0.5) * 255) << 16);
  kp->size = param.sigma * powf(2.0f, (layer + xi) / param.num_octave_layers) *
             (1 << octave) * 2;
  kp->response = abs(contr);

  return true;
}

float _calc_orientation_hist(const IMG_Mat &img, const SiftParam &param,
                             std::shared_ptr<KeyPoint> &kp, float scale, int n) {
  // -----------------------------计算描述子--------------------------------
  std::vector<float> &hist = kp->hist;
  hist.resize(n); // 初始化描述子
  int radius = static_cast<int>(param.ori_radius * scale + 0.5);

  int len = sqr(2 * radius + 1);

  float sigma = param.ori_sig_fctr * scale;

  float exp_scale = -1.0f / (2 * sigma * sigma);

  float *buffer = new float[4 * len + n + 4];
  float *X = buffer, *Y = buffer + len, *Mag = X, *Ori = Y + len,
        *W = Ori + len;
  float *temp_hist = W + len + 2;

  for (int i = 0; i < n; i++) {
    temp_hist[i] = 0.0f;
  }

  int k = 0;

  for (int i = -radius; i <= radius; i++) {
    int y = kp->y + i;

    if (y <= 0 || y >= img.rows - 1) {
      continue;
    }
    for (int j = -radius; j <= radius; j++) {
      int x = kp->x + j;

      if (x <= 0 || x >= img.cols - 1) {
        continue;
      }

      float dx = img.at<float>(y, x + 1) - img.at<float>(y, x - 1);
      float dy = img.at<float>(y - 1, x) - img.at<float>(y + 1, x);

      X[k] = dx;
      Y[k] = dy;
      W[k] = (sqr(i) + sqr(j)) * exp_scale;

      k++;
    }
  }

  len = k;

  for (int i = 0; i < len; i++) {
    // 计算领域内的所有像素的高斯权重
    W[i] = cv::exp(W[i]);
  }

  for (int i = 0; i < len; i++) {
    // 计算领域内所有像素的梯度方向
    Ori[i] = cv::fastAtan2(Y[i], X[i]);
  }
  cv::Mat _X = cv::Mat(1, len, CV_32F, X);
  cv::Mat _Y = cv::Mat(1, len, CV_32F, Y);
  cv::Mat _W = cv::Mat(1, len, CV_32F, W);
  // 计算领域内的所有像素的幅度
  cv::magnitude(_X, _Y, _W);
  for (int i = 0;i < len;i ++) {
    Mag[i] = _W.at<float>(0, i);
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
    max_val = std::max(max_val, hist[i]);
  }
  delete[] buffer;
  return max_val;
}

void _detect_keypoint(Image &src, const std::vector<Octave> &Octaves,
                      const SiftParam &param) {
  // -----------------------检测极值点--------------------------------
  LOG(INFO,
      "-----------------------检测极值点--------------------------------");
  int nOctaves = (int)Octaves.size();
  float threshold = (float)param.contrast_threshold / param.num_octave_layers;
  const int n = param.ori_hist_bins;
  float hist[n];

  src.keypoints.clear();
  int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1, 0};
  int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1, 0};

  for (int i = 0; i < param.num_octave; i++) {
    int numKeys = 0;
    for (int j = 1; j <= param.num_octave_layers; j++) {
      const IMG_Mat &curr_img = Octaves[i].dog_layers[j];
      const IMG_Mat &prev_img = Octaves[i].dog_layers[j - 1];
      const IMG_Mat &next_img = Octaves[i].dog_layers[j + 1];

      int num_row = curr_img.rows;
      int num_col = curr_img.cols;

      for (int x = param.img_border; x < num_row - param.img_border; x++) {
        for (int y = param.img_border; y < num_col - param.img_border; y++) {

          float val = curr_img.at<float>(x, y);
          float _00,_01,_02;
          float _10,    _12;
          float _20,_21,_22;

          float min_val,max_val;
          
          bool cond = std::abs(val) > threshold; 
          if (!(cond)) {
            continue;
          }

          // 检测极值点
          _00 = curr_img.at<float>(x - 1, y - 1);_01 = curr_img.at<float>(x - 1, y);_02 = curr_img.at<float>(x - 1, y + 1);
          _10 = curr_img.at<float>(x, y - 1);                                       _12 = curr_img.at<float>(x, y + 1);
          _20 = curr_img.at<float>(x + 1, y - 1);_21 = curr_img.at<float>(x + 1, y);_22 = curr_img.at<float>(x + 1, y + 1);       

          min_val = std::min({_00,_01,_02,_10,_12,_20,_21,_22});
          max_val = std::max({_00,_01,_02,_10,_12,_20,_21,_22});

          bool condp = cond & (val > 0) & (val >= max_val);
          bool condn = cond & (val < 0) & (val <= min_val);

          cond = condp | condn;

          if (!cond) {
            continue;
          }

          _00 = prev_img.at<float>(x - 1, y - 1);_01 = prev_img.at<float>(x - 1, y);_02 = prev_img.at<float>(x - 1, y + 1);
          _10 = prev_img.at<float>(x, y - 1);                                       _12 = prev_img.at<float>(x, y + 1); 
          _20 = prev_img.at<float>(x + 1, y - 1);_21 = prev_img.at<float>(x + 1, y);_22 = prev_img.at<float>(x + 1, y + 1);

          min_val = std::min({_00,_01,_02,_10,_12,_20,_21,_22});
          max_val = std::max({_00,_01,_02,_10,_12,_20,_21,_22});

          condp = (val >= max_val);
          condn = (val <= min_val);

          cond = condp | condn;
          if (!cond) {
            continue;
          }

          float _11p = prev_img.at<float>(x, y);
          float _11n = next_img.at<float>(x, y);

          float max_middle = std::max({_11p,_11n});
          float min_middle = std::min({_11p,_11n});

          _00 = next_img.at<float>(x - 1, y - 1);_01 = next_img.at<float>(x - 1, y);_02 = next_img.at<float>(x - 1, y + 1);
          _10 = next_img.at<float>(x, y - 1);                                       _12 = next_img.at<float>(x, y + 1);
          _20 = next_img.at<float>(x + 1, y - 1);_21 = next_img.at<float>(x + 1, y);_22 = next_img.at<float>(x + 1, y + 1);

          min_val = std::min({_00,_01,_02,_10,_12,_20,_21,_22});
          max_val = std::max({_00,_01,_02,_10,_12,_20,_21,_22});

          condp &= (val >= std::max(max_val, max_middle));
          condn &= (val <= std::min(min_val, min_middle));

          cond = condp | condn;
          if (!cond) {
            continue;
          }

          numKeys++;
          // 检测极值点
          std::shared_ptr<KeyPoint> kpt(new KeyPoint);
          int octave = i, layer = j, r1 = x, c1 = y;
          if (_adjust_local_extrema(src, Octaves, param, kpt, octave, layer,
                                    r1, c1)) {
            // ASSERT(kp.x == r1 && kp.y == c1,"kp(%d %d) != (%d %d)", kp.x,
            // kp.y, r1, c1);
            // LOG(INFO,"kpt(%d %d) != (%d %d)", kpt->x, kpt->y, r1, c1);
            float scale = (kpt->size * 0.5) / float(1 << octave);

            // 计算关键点的主方向
            float max_hist = _calc_orientation_hist(
                Octaves[octave].layers[layer], param, kpt, scale, n);

            float sum = 0.0;
            float mag_thr = 0.0;

            for (int i = 0; i < n; i++) {
              sum += kpt->hist[i];
            }
            mag_thr = (float)(max_hist * param.ori_peak_ratio);

            // 遍历所有bin
            for (int i = 0; i < n; i++) {
              int left = i > 0 ? i - 1 : n - 1;
              int right = i < n - 1 ? i + 1 : 0;

              //创建新的特征点，大于主峰的80%
              if (kpt->hist[i] > kpt->hist[left] &&
                  kpt->hist[i] > kpt->hist[right] &&
                  kpt->hist[i] >= mag_thr) {
                float bin =
                    i + 0.5f *
                            (kpt->hist[left] - kpt->hist[right]) /
                            (kpt->hist[left] + kpt->hist[right] -
                              2 * kpt->hist[i]);
                bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
                // 对主方向做一个细化
                float angle =360.0f - (360.0f / n) * bin;
                if (std::abs(angle - 360.0f) < 1e-7) {
                  angle = 0.0f;
                }
                kpt->angle = angle;
                src.keypoints.push_back(kpt);
              }
            }
          }
        }
      }
    }
    LOG(INFO, "numKeys = %d", numKeys);
    LOG(INFO, "detect point size : %ld", src.keypoints.size());
  }
}

void image_pyramid_create_self(Image &src, const SiftParam &param,
                               std::vector<Octave> &octaves) {
  LOG(INFO, "初始化第一张图像");
  IMG_Mat init_img;
  _init_first_img(src, param, init_img);
  std::vector<double> sigmas;
  LOG(INFO, "生成高斯模糊系数");
  _init_sigmas(param, sigmas);
  LOG(INFO, "生成图像金字塔");
  _init_octave_gauss_pyramid(src, param, sigmas, init_img, octaves);
  LOG(INFO, "生成差分图像");
  _init_dog_pyramid(src, octaves, param);
  LOG(INFO, "图像金字塔初始化完成");

  init_img.release();
  sigmas.clear();
}
void image_pyramid_create_opencv(Image &src, const SiftParam &param,
                                 std::vector<Octave> &octaves) {
  IMG_Mat gray_image;

  if (src.img.channels() != 1) {
    cv::cvtColor(src.img, gray_image, CV_RGB2GRAY);
  } else {
    gray_image = src.img.clone();
  }
  IMG_Mat floatImage, init_image;

  gray_image.convertTo(floatImage, CV_32FC1, 1.0 / 255.0, 0);
  double sig_diff = 0;

  if (param.keep_appearance) {
    IMG_Mat temp_image;
    cv::resize(floatImage, temp_image,
               cv::Size(2 * floatImage.cols, 2 * floatImage.rows), 0.0, 0.0,
               cv::INTER_LINEAR);

    sig_diff = sqrt(param.sigma * param.sigma -
                    4.0 * param.init_sigma * param.init_sigma);
    int kernel_width = 2 * round(param.gauss_kernel_patio * sig_diff) + 1;
    cv::Size kernel_size(kernel_width, kernel_width);

    cv::GaussianBlur(temp_image, init_image, kernel_size, sig_diff, sig_diff);
  } else {
    sig_diff = sqrt(param.sigma * param.sigma -
                    1.0 * param.init_sigma * param.init_sigma);

    int kernel_width = 2 * round(param.gauss_kernel_patio * sig_diff) + 1;
    cv::Size kernel_size(kernel_width, kernel_width);

    cv::GaussianBlur(floatImage, init_image, kernel_size, sig_diff, sig_diff);
  }

  std::vector<double> sig;
  sig.push_back(param.sigma);

  double k = pow(2.0, 1.0 / param.num_octave_layers);

  for (int i = 1; i < param.num_octave_layers + 3; ++i) {
    double prev_sig = pow(k, i - 1) * param.sigma; //每一个尺度层的尺度
    double curr_sig = k * prev_sig;

    //组内每层的尺度坐标计算公式
    sig.push_back(sqrt(curr_sig * curr_sig - prev_sig * prev_sig));
  }
  for (auto s : sig) {
    LOG(INFO, "sig = %f", s);
  }
  octaves.resize(param.num_octave);

  for (int i = 0; i < param.num_octave; i++) {
    octaves[i].layers.resize(param.num_octave_layers + 3);
  }

  for (int i = 0; i < param.num_octave; i++) {
    for (int j = 0; j < param.num_octave_layers + 3; j++) {
      if (i == 0 && j == 0) {
        octaves[i].layers[j] = init_image;
      } else if (j == 0) {
        cv::resize(octaves[i - 1].layers[3], octaves[i].layers[0],
                   cv::Size(octaves[i - 1].layers[3].cols / 2,
                            octaves[i - 1].layers[3].rows / 2),
                   0, 0, cv::INTER_LINEAR);
      } else {
        int kernel_width = 2 * round(param.gauss_kernel_patio * sig[j]) + 1;
        cv::Size kernel_size(kernel_width, kernel_width);

        cv::GaussianBlur(octaves[i].layers[j - 1], octaves[i].layers[j],
                         kernel_size, sig[j], sig[j]);
      }
     
    }
  }

  for (int i = 0; i < param.num_octave; i++) {
    for (int j = 0; j < param.num_octave_layers + 2; j++) {
      IMG_Mat temp_img = octaves[i].layers[j + 1] - octaves[i].layers[j];
      octaves[i].dog_layers.push_back(temp_img);
    }
  }
}
void _calc_sift_descriptor(const IMG_Mat &gauss_image,const SiftParam &param,float main_angle,
                           float x,float y,float scale,int d,int n,std::shared_ptr<KeyPoint> &kpt) {
  int kpt_x = int(x + 0.5);
  int kpt_y = int(y + 0.5);

  float cos_t = cosf(-main_angle * (float)(M_PI / 180.0));
  float sin_t = sinf(-main_angle * (float)(M_PI / 180.0));

  float bins_per_rad = n / 360.0f;

  float exp_scale = -1.0f / (d * d * 0.5f);

  float hist_width = param.descr_width * scale;

  int radius = int(hist_width * (d + 1) * sqrt(2) * 0.5f + 0.5);

  int rows = gauss_image.rows;
  int cols = gauss_image.cols;

  radius = std::min(radius, (int)sqrt((double)rows * rows + cols * cols));

  cos_t = cos_t / hist_width;
  sin_t = sin_t / hist_width;

  int len = (2 * radius + 1) * (2 * radius + 1);

  int histlen = (d + 2) * (d + 2) * (n + 2);

  float *buf = new float[6*len + histlen];
  float *X = buf, *Y = buf + len, *Mag = Y,*Angle = Y + len,*W = Angle + len;
  float *RBin = W + len,*CBin = RBin + len,*hist = CBin + len;

  for (int i = 0;i < d + 2;i ++) {
    for (int j = 0;j < d + 2;j ++) {
      for (int k = 0;k < n + 2;k ++) {
        hist[(i * (d + 2) + j) * (n + 2) + k] = 0.0f;
      }
    }
  }

  int k = 0;

  for (int i = -radius;i < radius;i ++) {
    for (int j = -radius;j < radius;j ++) {
      float c_rot = j * cos_t - i * sin_t;
      float r_rot = j * sin_t + i * cos_t;

      float cbin = c_rot + d / 2 - 0.5f;
      float rbin = r_rot + d / 2 - 0.5f;

      int r = kpt_x + i,c = kpt_y + j;

      if (rbin > -1 && rbin < d && cbin > -1 && cbin < d && r > 0 && r < rows - 1 && c > 0 && c < cols - 1) {
        float dx = gauss_image.at<float>(r, c + 1) - gauss_image.at<float>(r, c - 1);
        float dy = gauss_image.at<float>(r + 1, c) - gauss_image.at<float>(r - 1, c);

        X[k] = dx;
        Y[k] = dy;

        CBin[k] = cbin;
        RBin[k] = rbin;

        W[k] = (c_rot * c_rot + r_rot * r_rot) * exp_scale;

        k ++;
      } 
    }
  }

  len = k;

  for (int i = 0;i < len;i ++) {
    W[i] = exp(W[i]);
  }
  for (int i = 0;i < len;i ++) {
    Angle[i] = atan2f(Y[i], X[i]);
  }
  cv::Mat _X = cv::Mat(1, len, CV_32F, X);
  cv::Mat _Y = cv::Mat(1, len, CV_32F, Y);
  cv::Mat _W = cv::Mat(1, len, CV_32F);
  // 计算领域内的所有像素的幅度
  cv::magnitude(_X, _Y, _W);
  for (int i = 0;i < len;i ++) {
    Mag[i] = _W.at<float>(0, i);
  }

  for (k = 0;k < len;k ++) {
    float rbin = RBin[k];
    float cbin = CBin[k];

    float temp = Angle[k] - main_angle;

    float obin = temp * bins_per_rad;

    float mag = Mag[k] * W[k];

    int r0 = int(rbin + 0.5f);
    int c0 = int(cbin + 0.5f);
    int o0 = int(obin + 0.5f);

    rbin = rbin - r0;
    cbin = cbin - c0;
    obin = obin - o0;

    if (o0 < 0) {
      o0 += n;
    }
    if (o0 >= n) {
      o0 -= n;
    }

    float v_r1 = mag * rbin;
    float v_r0 = mag - v_r1;

    float v_rc11 = v_r1 * cbin;
    float v_rc10 = v_r1 - v_rc11;

    float v_rc01 = v_r0 * cbin;
    float v_rc00 = v_r0 - v_rc01;

    float v_rco111 = v_rc11 * obin;
    float v_rco110 = v_rc11 - v_rco111;

    float v_rco101 = v_rc10 * obin;
    float v_rco100 = v_rc10 - v_rco101;

    float v_rco011 = v_rc01 * obin;
    float v_rco010 = v_rc01 - v_rco011;

    float v_rco001 = v_rc00 * obin;
    float v_rco000 = v_rc00 - v_rco001;

    int idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
    hist[idx] += v_rco000;
    hist[idx + 1] += v_rco001;
    hist[idx + n + 2] += v_rco010;
    hist[idx + n + 3] += v_rco011;
    hist[idx + (d + 2) * (n + 2)] += v_rco100;
    hist[idx + (d + 2) * (n + 2) + 1] += v_rco101;
    hist[idx + (d + 3) * (n + 2)] += v_rco110;
    hist[idx + (d + 3) * (n + 2) + 1] += v_rco111;
  }

  for (int i = 0;i < d;i ++) {
    for (int j = 0;j < d;j ++) {
      int idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);
      
      hist[idx] += hist[idx + n];

      for (k = 0;k < n;k ++) {
        kpt->descriptor[(i * d + j) * n + k] = hist[idx + k];
      }
    }
  }

  int lenght = d * d * n;
  float norm = 0;

  for (int i = 0;i < lenght;i ++) {
    norm += kpt->descriptor[i] * kpt->descriptor[i];
  }
  norm = sqrt(norm);

  for (int i = 0;i < lenght;i ++) {
    kpt->descriptor[i] /= norm;
  }

  for (int i = 0;i < lenght;i ++) {
    kpt->descriptor[i] = std::min(kpt->descriptor[i], param.descr_mag_thr);
  }

  norm = 0;
  for (int i = 0;i < lenght;i ++) {
    norm += kpt->descriptor[i] * kpt->descriptor[i];
  }
  norm = sqrt(norm);
  for (int i = 0;i < lenght;i ++) {
    kpt->descriptor[i] /= norm;
  }

  delete[] buf;
}
void _calc_sift_descriptors(Image &src,std::vector<Octave> &octaves, const SiftParam &param) {
  LOG(INFO, "计算SIFT特征点的描述子");
  int d = param.descr_width;
  int n = param.descr_hist_bins;
  std::vector<std::shared_ptr<MY_IMG::KeyPoint>> &kpts = src.keypoints;
  for (int i = 0;i < kpts.size();i ++) {
    int octave = kpts[i]->octave & 255;
    int layer = (kpts[i]->octave >> 8) & 255;
    LOG(DEBUG, "octave: %d, layer: %d", octave, layer);

    int x = kpts[i]->x / (1 << octave);
    int y = kpts[i]->y / (1 << octave);
    LOG(DEBUG, "kpt.x: %d, kpt.y: %d", kpts[i]->x, kpts[i]->y);
    LOG(DEBUG, "x: %d, y: %d", x, y);

    float scale = kpts[i]->size / (1 << octave);
    float main_angle = kpts[i]->angle;
    kpts[i]->descriptor.resize(d * d * n);
    _calc_sift_descriptor(octaves[octave].layers[layer],param, main_angle, x, y, scale, d, n, kpts[i]);

    if (param.keep_appearance) {
      kpts[i]->x = kpts[i]->x / 2.0f;
      kpts[i]->y = kpts[i]->y / 2.0f;
    }
  }
}
void FeatureExtraction(Image &src, const SiftParam &param) {
  std::vector<Octave> octaves;
  // -----------------------初始化图像金字塔--------------------------------
  image_pyramid_create_opencv(src, param, octaves);
  // -----------------------获取关键点--------------------------------
  LOG(INFO, "检测关键点");
  _detect_keypoint(src, octaves, param);
  // 如果设置了特征点个数限制，则进行剪裁
  if (param.max_features != 0 && src.keypoints.size() > param.max_features) {
    std::sort(src.keypoints.begin(), src.keypoints.end(),
              [](const std::shared_ptr<KeyPoint> &a,
                 const std::shared_ptr<KeyPoint> &b) {
                return a->response > b->response;
              });
    // 删除多余的特征点
    src.keypoints.erase(src.keypoints.begin() + param.max_features,
                        src.keypoints.end());
  }
  _calc_sift_descriptors(src, octaves, param);
  // show
#ifdef DEBUG
  LOG(INFO, "keypoints size:%ld", src.keypoints.size());
  for (auto kpt : src.keypoints) {
    LOG(DEBUG, "keypoint: %d, %d, %f, %f, %f", kpt->x, kpt->y, kpt->size,
        kpt->angle, kpt->response);
  }
  int _o = 1;
  for (int i = 0; i < param.num_octave_layers + 3; i++) {
    cv::imshow("octave-" + std::to_string(_o) + "-layer-" + std::to_string(i),
               octaves[_o].layers[i]);
  }
  for (int i = 0; i < param.num_octave_layers + 2; i++) {
    cv::imshow("octave-" + std::to_string(_o) + "-dog-" + std::to_string(i),
               octaves[_o].dog_layers[i] * 255);
  }
  cv::waitKey(0);
#endif
  // -----------------------释放图像金字塔内存--------------------------------
  LOG(INFO, "释放图像金字塔内存");
  for (int i = 0; i < param.num_octave; i++) {
    for (int j = 0; j < param.num_octave_layers + 3; j++) {
      octaves.at(i).layers[j].release();
    }
    for (int j = 0; j < param.num_octave_layers + 2; j++) {
      octaves.at(i).dog_layers[j].release();
    }
    octaves.at(i).layers.clear();
    octaves.at(i).dog_layers.clear();
  }
  octaves.clear();
  LOG(INFO, "释放图像金字塔内存完成");
}

void SIFT(Image &src) {
  // -----------------------初始化参数--------------------------------
  LOG(INFO, "初始化sift参数");
  SiftParam param;
  int temp;
  float size_temp = (float)std::min(src.img.rows, src.img.cols);
  temp = int(std::log(size_temp) / std::log(2) - 2 + 0.5f);
  if (param.keep_appearance) {
    temp += 1;
  }
  if (temp > param.max_octave) {
    temp = param.max_octave;
  }
  param.num_octave = temp;
  FeatureExtraction(src, param);
}

} // namespace MY_IMG