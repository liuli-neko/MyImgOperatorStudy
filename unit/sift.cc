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
          src.Octaves.at(i).layers.at(j).size(), CV_32F);
      for (int x = 0; x < src.Octaves.at(i).layers.at(j).rows; ++x) {
        for (int y = 0; y < src.Octaves.at(i).layers.at(j).cols; ++y) {
          float tmp_val =
              static_cast<float>(
                  src.Octaves.at(i).layers.at(j).at<uint8_t>(x, y)) -
              src.Octaves.at(i).layers.at(j + 1).at<uint8_t>(x, y);
          src.Octaves.at(i).dog_layers.at(j).at<float>(x, y) = tmp_val;
        }
      }
      LOG("layers size : %d, %d", src.Octaves.at(i).layers.at(j).rows,
          src.Octaves.at(i).layers.at(j).cols);
    }
  }
}

void _detect_keypoint(Image &src, const SiftParam &param) {
  // -----------------------检测极值点--------------------------------

  // 通过比较每层的DoG值，检测极值点
  auto detect_keypoint = [&param](const IMG_Mat &m1, const IMG_Mat &m2,
                                  const IMG_Mat &m3,
                                  std::vector<Point2d> &keypoints) {
    bool max_val = true, min_val = true;
    int width = m2.cols, height = m2.rows;
    ASSERT(m1.rows == m2.rows && m1.cols == m2.cols && m1.rows == m3.rows &&
               m1.cols == m3.cols,
           "图像大小不一致");

    int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1, 0};
    int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1, 0};

    for (int i = 1; i < height - 1 && (max_val || min_val); ++i) {
      for (int j = 1; j < width - 1 && (max_val || min_val); ++j) {
        for (int x = 0; x < 9; ++x) {
          if (m2.at<float>(i, j) < m1.at<float>(i + dx[x], j + dy[x])) {
            max_val = false;
          }
          if (m2.at<float>(i, j) < m2.at<float>(i + dx[x], j + dy[x])) {
            max_val = false;
          }
          if (m2.at<float>(i, j) < m3.at<float>(i + dx[x], j + dy[x])) {
            max_val = false;
          }
          if (m2.at<float>(i, j) > m1.at<float>(i + dx[x], j + dy[x])) {
            min_val = false;
          }
          if (m2.at<float>(i, j) > m2.at<float>(i + dx[x], j + dy[x])) {
            min_val = false;
          }
          if (m2.at<float>(i, j) > m3.at<float>(i + dx[x], j + dy[x])) {
            min_val = false;
          }
        }
        if (max_val || min_val) {
          // 计算Harris角点，删除不合格的点，这里为什么没有算尺度维？
          float D_xx = static_cast<float>(m2.at<float>(i, j + 1)) -
                       2.0 * m1.at<float>(i, j) + m2.at<float>(i, j - 1);
          float D_yy = static_cast<float>(m2.at<float>(i + 1, j)) -
                       2.0 * m1.at<float>(i, j) + m2.at<float>(i - 1, j);
          float D_xy = static_cast<float>(m2.at<float>(i + 1, j + 1)) -
                       m2.at<float>(i + 1, j) - m2.at<float>(i, j + 1) +
                       m2.at<float>(i, j);
          if ((D_xx + D_yy) * (D_xx + D_yy) / (D_xx * D_yy - D_xy * D_xy) <
              (param.Hessian_r + 1) * (param.Hessian_r + 1) / param.Hessian_r) {
            keypoints.push_back(Point2d(j, i));
          }
        }
      }
    }
  };

  // 对每层的DoG值进行检测
  for (int i = 0; i < param.octave_num; ++i) {
    std::vector<Point2d> keypoints;
    keypoints.clear();
    for (int j = 0; j < param.octave_layer_num - 3; ++j) {
      detect_keypoint(src.Octaves.at(i).dog_layers.at(j),
                      src.Octaves.at(i).dog_layers.at(j + 1),
                      src.Octaves.at(i).dog_layers.at(j + 2), keypoints);
    }
    // 除了第一层，其他层的点都有坐标变换问题呢，得还原到输入的尺度？
    // 先去重吧
    std::sort(keypoints.begin(), keypoints.end());
    keypoints.erase(std::unique(keypoints.begin(), keypoints.end()),
                    keypoints.end());
    LOG("keypoints size : %ld", keypoints.size());
    LOG("当前层：%d", i);
    // 转换缩放
    for (auto &p : keypoints) {
      LOG("(%d, %d) -> (%.0lf ,%.0lf)", p.x, p.y, p.x * pow(param.octave_k, i),
          p.y * pow(param.octave_k, i));
      src.keypoints.push_back({static_cast<int>(p.x * pow(param.octave_k, i)),
                               static_cast<int>(p.y * pow(param.octave_k, i))});
    }
  }

  // 去除重复的点
  std::sort(src.keypoints.begin(), src.keypoints.end());
  src.keypoints.erase(std::unique(src.keypoints.begin(), src.keypoints.end()),
                      src.keypoints.end());
  LOG("keypoints size : %ld", src.keypoints.size());
}

void SIFT(Image &src, std::vector<SiftPointDescriptor> &descriptors) {
  // -----------------------初始化参数--------------------------------
  LOG("初始化sift参数");
  SiftParam param;
  param.octave_num =
      static_cast<int>(log2(std::min(src.img.rows, src.img.cols))) - 3;
  param.octave_layer_num = 6;
  param.octave_layer_min_sigma = 1.6;
  param.octave_layer_max_sigma = 3.2;
  param.octave_layer_sigma_k = 1.0 / param.octave_layer_num;

  // -----------------------初始化图像金字塔--------------------------------
  LOG("初始化图像金字塔");
  _init_octave(src, param);

  // -----------------------检测关键点--------------------------------
  LOG("检测关键点");
  _detect_keypoint(src, param);

  // -----------------------计算描述子--------------------------------

  // -----------------------show--------------------------------
  // show

  LOG("keyPoints size : %ld", src.keypoints.size());
  for (auto &p : src.keypoints) {
    LOG("(%d, %d)", p.x, p.y);
  }

  int _i = 0;
  cv::imshow("src0", src.Octaves.at(_i).layers.at(0));
  IMG_Mat temp;

  for (int j = 1; j < param.octave_layer_num; ++j) {
    cv::imshow("layer" + std::to_string(j), src.Octaves.at(_i).layers.at(j));
    temp = IMG_Mat(src.Octaves.at(_i).dog_layers.at(j - 1).size(), CV_8U);
    double sum = cv::sum(src.Octaves.at(_i).dog_layers.at(j - 1))[0];
    for (int x = 0; x < src.Octaves.at(_i).dog_layers.at(j - 1).rows; ++x) {
      for (int y = 0; y < src.Octaves.at(_i).dog_layers.at(j - 1).cols; ++y) {
        float val =
            (src.Octaves.at(_i).dog_layers.at(j - 1).at<float>(x, y) / sum) *
            255;
        if (val < 0)
          val = -val;
        if (val > 255)
          val = 255;
        temp.at<uchar>(x, y) = val;
      }
    }
    cv::imshow("dog" + std::to_string(j), temp);
  }
  IMG_Mat keypoints_img;
  MY_IMG::DrawPoints(src.img, src.keypoints, keypoints_img);
  cv::imshow("keypoints", keypoints_img);

  cv::waitKey(0);
}

} // namespace MY_IMG