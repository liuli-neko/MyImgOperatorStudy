#include "mySift.h"
#include "all.h"
#include "image_operator.h"

#include <algorithm>
#include <cmath>
#include <iostream> //输入输出
#include <numeric>  //用于容器元素求和
#include <string>
#include <vector> //vector

#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/hal/intrin.hpp>

#include <opencv2/core/core.hpp>             //opencv基本数据结构
#include <opencv2/features2d/features2d.hpp> //特征提取
#include <opencv2/highgui/highgui.hpp>       //图像界面
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> //基本图像处理函数
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

/******************根据输入图像大小计算高斯金字塔的组数****************************/
/*image表示原始输入灰度图像,inline函数必须在声明处定义
double_size_image表示是否在构建金字塔之前上采样原始图像
*/
int mySift::num_octaves(const Mat &image) {
  int temp;
  float size_temp = (float)min(image.rows, image.cols);
  temp = cvRound(log(size_temp) / log((float)2) - 2);

  if (double_size)
    temp += 1;
  if (temp > MAX_OCTAVES) //尺度空间最大组数设置为MAX_OCTAVES
    temp = MAX_OCTAVES;

  return temp;
}

/************************sobel滤波器计算高斯尺度空间图像梯度大小*****************************/
void sobelfilter(Mat &image, Mat &G) {
  // image 是经过归一化后的数据 (0,1)
  int rows = image.rows;
  int cols = image.cols;

  float dx = 0.0, dy = 0.0;

  // cv::Mat Gx = cv::Mat::zeros(rows, cols, CV_32FC1);
  // //包含了图像像素在水平方向上的导数的近似值的图像 cv::Mat Gy =
  // cv::Mat::zeros(rows, cols, CV_32FC1);
  // //包含了图像像素在垂直方向上的导数的近似值的图像

  G = Mat::zeros(rows, cols,
                 CV_32FC1); //在每个像素点处的灰度大小由Gx和Gy共同决定

  double v = 0.0, vx, vy;

  //利用 sobel 算子求梯度幅度图像
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      v = 0.0;

      if (i == 0 || j == 0 || i == rows - 1 || j == cols - 1) {
        G.at<float>(i, j) = 0.0;
      } else {
        float dx =
            image.at<float>(i - 1, j + 1) - image.at<float>(i - 1, j - 1) +
            2 * image.at<float>(i, j + 1) - 2 * image.at<float>(i, j - 1) +
            image.at<float>(i + 1, j + 1) - image.at<float>(i + 1, j - 1);

        float dy =
            image.at<float>(i + 1, j - 1) - image.at<float>(i - 1, j - 1) +
            2 * image.at<float>(i + 1, j) - 2 * image.at<float>(i - 1, j) +
            image.at<float>(i + 1, j + 1) - image.at<float>(i - 1, j + 1);

        v = abs(dx) + abs(dy); //简化后 G = |Gx| + |Gy|

        //保证像素值在有效范围内
        v = fmax(v, 0);   //返回浮点数中较大的一个
        v = fmin(v, 255); //返回浮点数中较小的一个

        if (v > T) // T为阈值等于50
          G.at<float>(i, j) = (float)v;
        else
          G.at<float>(i, j) = 0.0;
      }
    }
  }

  //水平方向上的导数的近似值的图像
  /*for (int i = 0; i < rows; i++)
  {
          for (int j = 0; j < cols; j++)
          {
                  vx = 0;
                  if (i == 0 || j == 0 || i == rows - 1 || j == cols - 1)
                          Gx.at<float>(i, j) = 0;
                  else
                  {
                          dx = image.at<float>(i - 1, j + 1) - image.at<float>(i
  - 1, j - 1)
                                  + 2 * image.at<float>(i, j + 1) - 2 *
  image.at<float>(i, j - 1)
                                  + image.at<float>(i + 1, j + 1) -
  image.at<float>(i + 1, j - 1); vx = abs(dx); vx = fmax(vx, 0); vx = fmin(vx,
  255); Gx.at<float>(i, j) = (float)vx;
                  }
          }
  }*/

  //垂直方向上的导数的近似值的图像
  /*for (int i = 0; i < rows; i++)
  {
          for (int j = 0; j < cols; j++)
          {
                  vy = 0;
                  if (i == 0 || j == 0 || i == rows - 1 || j == cols - 1)
                          Gy.at<float>(i, j) = 0;
                  else
                  {
                          dy = image.at<float>(i + 1, j - 1) - image.at<float>(i
  - 1, j - 1)
                                  + 2 * image.at<float>(i + 1, j) - 2 *
  image.at<float>(i - 1, j) + image.at<float>(i + 1, j + 1) - image.at<float>(i
  - 1, j + 1); vy = abs(dy); vy = fmax(vy, 0); vx = fmin(vy, 255);
                          Gy.at<float>(i, j) = (float)vy;
                  }
          }
  }*/

  // cv::imshow("Gx", Gx);						//
  // horizontal
  // cv::imshow("Gy", Gy);						//
  // vertical
  // cv::imshow("G", G);						//
  // gradient
}

/*********该函数根据尺度和窗口半径生成ROEWA滤波模板************/
/*size表示核半径，因此核宽度是2*size+1
 scale表示指数权重参数
 kernel表示生成的滤波核
 */
static void roewa_kernel(int size, float scale, Mat &kernel) {
  kernel.create(2 * size + 1, 2 * size + 1, CV_32FC1);
  for (int i = -size; i <= size; ++i) {
    float *ptr_k = kernel.ptr<float>(i + size);
    for (int j = -size; j <= size; ++j) {
      ptr_k[j + size] = exp(-1.f * (abs(i) + abs(j)) / scale);
    }
  }
}

/************************创建高斯金字塔第一组，第一层图像************************************/
/*image表示输入原始图像
 init_image表示生成的高斯尺度空间的第一层图像
 */
void mySift::create_initial_image(const Mat &image, Mat &init_image) {
  Mat gray_image;

  if (image.channels() != 1)
    cvtColor(image, gray_image, CV_RGB2GRAY); //转换为灰度图像
  else
    gray_image = image.clone();

  Mat floatImage; //转换到0-1之间的浮点类型数据归一化，方便接下来的处理

  // float_image=(float)gray_image*(1.0/255.0)
  gray_image.convertTo(floatImage, CV_32FC1, 1.0 / 255.0, 0);

  double sig_diff = 0;

  if (double_size) {
    Mat temp_image;

    //通过插值的方法改变图像尺寸的大小
    resize(floatImage, temp_image,
           Size(2 * floatImage.cols, 2 * floatImage.rows), 0.0, 0.0,
           INTER_LINEAR);

    //高斯平滑的标准差，值较大时平滑效果比较明显
    sig_diff = sqrt(sigma * sigma - 4.0 * INIT_SIGMA * INIT_SIGMA);

    //高斯滤波窗口大小选择很重要，这里选择(4*sig_diff_1+1)-(6*sig_diff+1)之间，且四舍五入
    int kernel_width = 2 * cvRound(GAUSS_KERNEL_RATIO * sig_diff) + 1;

    Size kernel_size(kernel_width, kernel_width);

    //对图像进行平滑处理(高斯模糊)，即降低图像的分辨率,高斯模糊是实现尺度变换的唯一变换核，并其实唯一的线性核
    GaussianBlur(temp_image, init_image, kernel_size, sig_diff, sig_diff);
  } else {
    sig_diff = sqrt(sigma * sigma - 1.0 * INIT_SIGMA * INIT_SIGMA);

    //高斯滤波窗口大小选择很重要，这里选择(4*sig_diff_1+1)-(6*sig_diff+1)之间
    int kernel_width = 2 * cvRound(GAUSS_KERNEL_RATIO * sig_diff) + 1;

    Size kernel_size(kernel_width, kernel_width);

    GaussianBlur(floatImage, init_image, kernel_size, sig_diff, sig_diff);
  }
}

/************************使用 sobel
 * 算子创建高斯金字塔第一组，第一层图像****************************/
//目的是为了减少冗余特征点
void mySift::sobel_create_initial_image(const Mat &image, Mat &init_image) {
  Mat gray_image, gray_images; // gray_images用于存放经过sobel算子操作后的图像

  if (image.channels() != 1)
    cvtColor(image, gray_image, CV_RGB2GRAY); //转换为灰度图像
  else
    gray_image = image.clone();

  sobelfilter(gray_image, gray_images);

  Mat floatImage; //转换到0-1之间的浮点类型数据归一化，方便接下来的处理

  // float_image=(float)gray_image*(1.0/255.0)
  gray_images.convertTo(floatImage, CV_32FC1, 1.0 / 255.0, 0);

  double sig_diff = 0;

  if (double_size) {
    Mat temp_image;

    //通过插值的方法改变图像尺寸的大小
    resize(floatImage, temp_image,
           Size(2 * floatImage.cols, 2 * floatImage.rows), 0.0, 0.0,
           INTER_LINEAR);

    //高斯平滑的标准差，值较大时平滑效果比较明显
    sig_diff = sqrt(sigma * sigma - 4.0 * INIT_SIGMA * INIT_SIGMA);

    //高斯滤波窗口大小选择很重要，这里选择(4*sig_diff_1+1)-(6*sig_diff+1)之间，且四舍五入
    int kernel_width = 2 * cvRound(GAUSS_KERNEL_RATIO * sig_diff) + 1;

    Size kernel_size(kernel_width, kernel_width);

    //对图像进行平滑处理(高斯模糊)，即降低图像的分辨率,高斯模糊是实现尺度变换的唯一变换核，并其实唯一的线性核
    GaussianBlur(temp_image, init_image, kernel_size, sig_diff, sig_diff);
  } else {
    sig_diff = sqrt(sigma * sigma - 1.0 * INIT_SIGMA * INIT_SIGMA);

    //高斯滤波窗口大小选择很重要，这里选择(4*sig_diff_1+1)-(6*sig_diff+1)之间
    int kernel_width = 2 * cvRound(GAUSS_KERNEL_RATIO * sig_diff) + 1;

    Size kernel_size(kernel_width, kernel_width);

    GaussianBlur(floatImage, init_image, kernel_size, sig_diff, sig_diff);
  }
}

/**************************生成高斯金字塔*****************************************/
/*init_image表示已经生成的高斯金字塔第一层图像
 gauss_pyramid表示生成的高斯金字塔
 nOctaves表示高斯金字塔的组数
*/
void mySift::build_gaussian_pyramid(const Mat &init_image,
                                    vector<vector<Mat>> &gauss_pyramid,
                                    int nOctaves) {
  vector<double> sig;

  sig.push_back(sigma);

  double k = pow(2.0, 1.0 / nOctaveLayers); //高斯金字塔每一层的系数 k

  for (int i = 1; i < nOctaveLayers + 3; ++i) {
    double prev_sig = pow(k, i - 1) * sigma; //每一个尺度层的尺度
    double curr_sig = k * prev_sig;

    //组内每层的尺度坐标计算公式
    sig.push_back(sqrt(curr_sig * curr_sig - prev_sig * prev_sig));
  }
  for (auto s : sig) {
    LOG(INFO,"sig:%lf", s);
  }
  gauss_pyramid.resize(nOctaves);

  for (int i = 0; i < nOctaves; ++i) {
    gauss_pyramid[i].resize(nOctaveLayers + 3);
  }

  for (int i = 0; i < nOctaves; ++i) //对于每一组
  {
    for (int j = 0; j < nOctaveLayers + 3; ++j) //对于组内的每一层
    {
      if (i == 0 && j == 0) //第一组，第一层
        gauss_pyramid[0][0] = init_image;

      else if (j == 0) {
        resize(gauss_pyramid[i - 1][3], gauss_pyramid[i][0],
               Size(gauss_pyramid[i - 1][3].cols / 2,
                    gauss_pyramid[i - 1][3].rows / 2),
               0, 0, INTER_LINEAR);
      } else {
        //高斯滤波窗口大小选择很重要，这里选择(4*sig_diff_1+1)-(6*sig_diff+1)之间
        int kernel_width = 2 * cvRound(GAUSS_KERNEL_RATIO * sig[j]) + 1;
        Size kernel_size(kernel_width, kernel_width);

        GaussianBlur(gauss_pyramid[i][j - 1], gauss_pyramid[i][j], kernel_size,
                     sig[j], sig[j]);
      }
    }
  }
}

/*******************生成高斯差分金字塔，即LOG金字塔*************************/
/*dog_pyramid表示DOG金字塔
 gauss_pyramin表示高斯金字塔*/
void mySift::build_dog_pyramid(vector<vector<Mat>> &dog_pyramid,
                               const vector<vector<Mat>> &gauss_pyramid) {
  vector<vector<Mat>>::size_type nOctaves = gauss_pyramid.size();

  for (vector<vector<Mat>>::size_type i = 0; i < nOctaves; ++i) {
    //用于存放每一个梯度中的所有尺度层
    vector<Mat> temp_vec;

    for (auto j = 0; j < nOctaveLayers + 2; ++j) {
      Mat temp_img = gauss_pyramid[i][j + 1] - gauss_pyramid[i][j];

      temp_vec.push_back(temp_img);
    }
    dog_pyramid.push_back(temp_vec);

    temp_vec.clear();
  }
}

/***********生成高斯差分金字塔当前层对应的梯度幅度图像和梯度方向图像***********/
/*image为高斯差分金字塔当前层图像
 *amplit为当前层梯度幅度图像
 *orient为当前层梯度方向图像
 *scale当前层尺度
 *nums为相对底层的层数
 */
void mySift::amplit_orient(const Mat &image, vector<Mat> &amplit,
                           vector<Mat> &orient, float scale, int nums) {
  //分配内存
  amplit.resize(Mmax * nOctaveLayers);
  orient.resize(Mmax * nOctaveLayers);

  int radius = cvRound(2 * scale);

  Mat kernel; // kernel(2 * radius + 1, 2 * radius + 1, CV_32FC1);

  roewa_kernel(radius, scale,
               kernel); //返回滤波核，也即指数部分,存放在矩阵的右下角

  //四个滤波模板生成
  Mat W34 = Mat::zeros(2 * radius + 1, 2 * radius + 1,
                       CV_32FC1); //把kernel矩阵下半部分复制到对应部分
  Mat W12 = Mat::zeros(2 * radius + 1, 2 * radius + 1,
                       CV_32FC1); //把kernel矩阵上半部分复制到对应部分
  Mat W14 = Mat::zeros(2 * radius + 1, 2 * radius + 1,
                       CV_32FC1); //把kernel矩阵右半部分复制到对应部分
  Mat W23 = Mat::zeros(2 * radius + 1, 2 * radius + 1,
                       CV_32FC1); //把kernel矩阵左半部分复制到对应部分

  kernel(Range(radius + 1, 2 * radius + 1), Range::all())
      .copyTo(W34(Range(radius + 1, 2 * radius + 1), Range::all()));
  kernel(Range(0, radius), Range::all())
      .copyTo(W12(Range(0, radius), Range::all()));
  kernel(Range::all(), Range(radius + 1, 2 * radius + 1))
      .copyTo(W14(Range::all(), Range(radius + 1, 2 * radius + 1)));
  kernel(Range::all(), Range(0, radius))
      .copyTo(W23(Range::all(), Range(0, radius)));

  //滤波
  Mat M34, M12, M14, M23;
  double eps = 0.00001;

  // float_image为图像归一化后的图像数据，做卷积运算
  filter2D(image, M34, CV_32FC1, W34, Point(-1, -1), eps);
  filter2D(image, M12, CV_32FC1, W12, Point(-1, -1), eps);
  filter2D(image, M14, CV_32FC1, W14, Point(-1, -1), eps);
  filter2D(image, M23, CV_32FC1, W23, Point(-1, -1), eps);

  //计算水平梯度和竖直梯度
  Mat Gx, Gy;
  log((M14) / (M23), Gx);
  log((M34) / (M12), Gy);

  //计算梯度幅度和梯度方向
  magnitude(Gx, Gy, amplit[nums]);   //梯度幅度图像，平方和开平方
  phase(Gx, Gy, orient[nums], true); //梯度方向图像
}

/***********************该函数计算尺度空间特征点的主方向，用于后面特征点的检测***************************/
/*image表示特征点所在位置的高斯图像，后面可对着源码进行修改
 pt表示特征点的位置坐标(x,y)
 scale特征点的尺度
 n表示直方图bin个数
 hist表示计算得到的直方图
 函数返回值是直方图hist中的最大数值*/
static float clac_orientation_hist(const Mat &image, Point pt, float scale,
                                   int n, float *hist) {
  int radius = cvRound(ORI_RADIUS * scale); //特征点邻域半径(3*1.5*scale)

  int len =
      (2 * radius + 1) * (2 * radius + 1); //特征点邻域像素总个数（最大值）

  float sigma = ORI_SIG_FCTR * scale; //特征点邻域高斯权重标准差(1.5*scale)

  float exp_scale = -1.f / (2 * sigma * sigma); //权重的指数部分

  //使用AutoBuffer分配一段内存，这里多出4个空间的目的是为了方便后面平滑直方图的需要
  AutoBuffer<float> buffer((4 * len) + n + 4);

  // X保存水平差分，Y保存数值差分，Mag保存梯度幅度，Ori保存梯度角度，W保存高斯权重
  float *X = buffer, *Y = buffer + len, *Mag = Y, *Ori = Y + len,
        *W = Ori + len;
  float *temp_hist = W + len + 2; //临时保存直方图数据

  for (int i = 0; i < n; ++i)
    temp_hist[i] = 0.f; //数据清零

  //计算邻域像素的水平差分和竖直差分
  int k = 0;

  for (int i = -radius; i < radius; ++i) {
    int y = pt.y + i; //邻域点的纵坐标

    if (y <= 0 || y >= image.rows - 1)
      continue;

    for (int j = -radius; j < radius; ++j) {
      int x = pt.x + j;

      if (x <= 0 || x >= image.cols - 1)
        continue;

      float dx =
          image.at<float>(y, x + 1) - image.at<float>(y, x - 1); //水平差分
      float dy =
          image.at<float>(y + 1, x) - image.at<float>(y - 1, x); //竖直差分

      //保存水平差分和竖直差分及其对应的权重
      X[k] = dx;
      Y[k] = dy;
      W[k] = (i * i + j * j) * exp_scale;

      ++k;
    }
  }

  len = k; //邻域内特征点的个数

  cv::hal::exp(W, W, len); //计算邻域内所有像素的高斯权重

  cv::hal::fastAtan2(Y, X, Ori, len,
                     true); //计算邻域内所有像素的梯度方向，角度范围0-360度
  // for (int i = 0;i < len;i ++) {
  //   LOG(INFO,"Ori[%d] = %f , %f", i, Ori[i],cv::fastAtan2(Y[i], X[i]));
  // }
  cv::hal::magnitude32f(
      X, Y, Mag, len); //计算邻域内所有像素的梯度幅度，计算的是数学意义上的梯度
  // for (int i = 0;i < len;i ++) {
  //   LOG(INFO,"Mag[%d] = %f, %f", i, Mag[i],sqrt(X[i] * X[i] + Y[i] * Y[i]));
  // }

  //遍历邻域的像素
  for (int i = 0; i < len; ++i) {
    int bin = cvRound((n / 360.f) *
                      Ori[i]); //利用像素的梯度方向，约束bin的范围在[0,(n-1)]

    //像素点梯度方向为360度时，和0°一样
    if (bin >= n)
      bin = bin - n;

    if (bin < 0)
      bin = bin + n;

    temp_hist[bin] =
        temp_hist[bin] +
        Mag[i] * W[i]; //统计邻域内像素各个方向在梯度直方图的幅值(加权后的幅值)
  }

  //平滑直方图
  temp_hist[-1] = temp_hist[n - 1];
  temp_hist[-2] = temp_hist[n - 2];
  temp_hist[n] = temp_hist[0];
  temp_hist[n + 1] = temp_hist[1];
  for (int i = 0; i < n; ++i) {
    hist[i] = (temp_hist[i - 2] + temp_hist[i + 2]) * (1.f / 16.f) +
              (temp_hist[i - 1] + temp_hist[i + 1]) * (4.f / 16.f) +
              temp_hist[i] * (6.f / 16.f);
  }

  //获得直方图中最大值
  float max_value = hist[0];
  for (int i = 1; i < n; ++i) {
    if (hist[i] > max_value)
      max_value = hist[i];
  }
  return max_value;
}

/***********************使用 sobel
 * 滤波器定义的新梯度计算尺度空间特征点的主方向**************************/
static float clac_orientation_hist_2(Mat &image, Point pt, float scale, int n,
                                     float *hist) {
  Mat output_image; //使用 sobel 滤波器计算的图像的梯度幅度图像

  sobelfilter(image,
              output_image); //使用 sobel 滤波器求高斯差分图像的梯度幅度图像

  int radius = cvRound(ORI_RADIUS * scale); //特征点邻域半径(3*1.5*scale)

  int len =
      (2 * radius + 1) * (2 * radius + 1); //特征点邻域像素总个数（最大值）

  float sigma = ORI_SIG_FCTR * scale; //特征点邻域高斯权重标准差(1.5*scale)

  float exp_scale = -1.f / (2 * sigma * sigma); //权重的指数部分

  //使用AutoBuffer分配一段内存，这里多出4个空间的目的是为了方便后面平滑直方图的需要
  AutoBuffer<float> buffer((4 * len) + n + 4);

  // X保存水平差分，Y保存数值差分，Mag保存梯度幅度，Ori保存梯度角度，W保存高斯权重
  float *X = buffer, *Y = buffer + len, *Mag = Y, *Ori = Y + len,
        *W = Ori + len;
  float *temp_hist = W + len + 2; //临时保存直方图数据

  for (int i = 0; i < n; ++i)
    temp_hist[i] = 0.f; //数据清零

  //计算邻域像素的水平差分和竖直差分
  int k = 0;

  for (int i = -radius; i < radius; ++i) {
    int y = pt.y + i; //邻域点的纵坐标，行

    if (y <= 0 || y >= output_image.rows - 1)
      continue;

    for (int j = -radius; j < radius; ++j) {
      int x = pt.x + j; //邻域点的纵坐标，列

      if (x <= 0 || x >= output_image.cols - 1)
        continue;

      // float dx = image.at<float>(y, x + 1) - image.at<float>(y, x - 1);
      // //水平差分

      float dx = output_image.at<float>(y - 1, x + 1) -
                 output_image.at<float>(y - 1, x - 1) +
                 2 * output_image.at<float>(y, x + 1) -
                 2 * output_image.at<float>(y, x - 1) +
                 output_image.at<float>(y + 1, x + 1) -
                 output_image.at<float>(y + 1, x - 1);

      float dy = output_image.at<float>(y + 1, x - 1) -
                 output_image.at<float>(y - 1, x - 1) +
                 2 * output_image.at<float>(y + 1, x) -
                 2 * output_image.at<float>(y - 1, x) +
                 output_image.at<float>(y + 1, x + 1) -
                 output_image.at<float>(y - 1, x + 1);

      /*float dx = image.at<float>(y - 1, x + 1) - image.at<float>(y - 1, x - 1)
              + 2 * image.at<float>(y, x + 1) - 2 * image.at<float>(y, x - 1)
              + image.at<float>(y + 1, x + 1) - image.at<float>(y + 1, x - 1);
      float dy = image.at<float>(y + 1, x - 1) - image.at<float>(y - 1, x - 1)
              + 2 * image.at<float>(y + 1, x) - 2 * image.at<float>(y - 1, x) +
              image.at<float>(y + 1, x + 1) - image.at<float>(y - 1, x + 1);*/

      //保存水平差分和竖直差分及其对应的权重
      X[k] = dx;
      Y[k] = dy;
      W[k] = (i * i + j * j) * exp_scale;

      ++k;
    }
  }

  len = k; //邻域内特征点的个数

  cv::hal::exp(W, W, len); //计算邻域内所有像素的高斯权重
  cv::hal::fastAtan2(Y, X, Ori, len,
                     true); //计算邻域内所有像素的梯度方向，角度范围0-360度
  cv::hal::magnitude32f(
      X, Y, Mag, len); //计算邻域内所有像素的梯度幅度，计算的是数学意义上的梯度

  //遍历邻域的像素
  for (int i = 0; i < len; ++i) {
    int bin = cvRound((n / 360.f) *
                      Ori[i]); //利用像素的梯度方向，约束bin的范围在[0,(n-1)]

    //像素点梯度方向为360度时，和0°一样
    if (bin >= n)
      bin = bin - n;

    if (bin < 0)
      bin = bin + n;

    temp_hist[bin] =
        temp_hist[bin] +
        Mag[i] * W[i]; //统计邻域内像素各个方向在梯度直方图的幅值(加权后的幅值)
  }

  //平滑直方图
  temp_hist[-1] = temp_hist[n - 1];
  temp_hist[-2] = temp_hist[n - 2];
  temp_hist[n] = temp_hist[0];
  temp_hist[n + 1] = temp_hist[1];
  for (int i = 0; i < n; ++i) {
    hist[i] = (temp_hist[i - 2] + temp_hist[i + 2]) * (1.f / 16.f) +
              (temp_hist[i - 1] + temp_hist[i + 1]) * (4.f / 16.f) +
              temp_hist[i] * (6.f / 16.f);
  }

  //获得直方图中最大值
  float max_value = hist[0];
  for (int i = 1; i < n; ++i) {
    if (hist[i] > max_value)
      max_value = hist[i];
  }
  return max_value;
}

/******************该函数计算特征点主方向,用于LOGH版本*********************/
/*amplit表示特征点所在层的梯度幅度，即输入图像对应像素点的梯度存在了对应位置
  orient表示特征点所在层的梯度方向，0-360度
  point表示特征点坐标
  scale表示特征点的所在层的尺度
  hist表示生成的直方图
  n表示主方向直方图bin个数
 */
static float calc_orient_hist(const Mat &amplit, const Mat &orient, Point pt,
                              float scale, float *hist, int n) {
  //暂且认为是只进行了下采样，没有进行高斯模糊
  int num_row = amplit.rows;
  int num_col = amplit.cols;

  Point point(cvRound(pt.x), cvRound(pt.y));

  // int radius = cvRound(SAR_SIFT_FACT_RADIUS_ORI * scale);
  int radius = cvRound(6 * scale);

  radius = min(radius, min(num_row / 2, num_col / 2));

  float gauss_sig = 2 * scale; //高斯加权标准差

  float exp_temp = -1.f / (2 * gauss_sig * gauss_sig); //权重指数部分

  //邻域区域
  int radius_x_left = point.x - radius;
  int radius_x_right = point.x + radius;
  int radius_y_up = point.y - radius;
  int radius_y_down = point.y + radius;

  //防止越界
  if (radius_x_left < 0)
    radius_x_left = 0;
  if (radius_x_right > num_col - 1)
    radius_x_right = num_col - 1;
  if (radius_y_up < 0)
    radius_y_up = 0;
  if (radius_y_down > num_row - 1)
    radius_y_down = num_row - 1;

  //此时特征点周围矩形区域相对于本矩形区域的中心
  int center_x = point.x - radius_x_left;
  int center_y = point.y - radius_y_up;

  //矩形区域的边界，计算高斯权值
  Range x_rng(-(point.x - radius_x_left), radius_x_right - point.x);
  Range y_rng(-(point.y - radius_y_up), radius_y_down - point.y);

  Mat gauss_weight;

  gauss_weight.create(y_rng.end - y_rng.start + 1, x_rng.end - x_rng.start + 1,
                      CV_32FC1);

  //求各个像素点的高斯权重
  for (int i = y_rng.start; i <= y_rng.end; ++i) {
    float *ptr_gauss = gauss_weight.ptr<float>(i - y_rng.start);

    for (int j = x_rng.start; j <= x_rng.end; ++j)
      ptr_gauss[j - x_rng.start] = exp((i * i + j * j) * exp_temp);
  }

  //索引特征点周围的像素梯度幅度，梯度方向
  Mat sub_amplit = amplit(Range(radius_y_up, radius_y_down + 1),
                          Range(radius_x_left, radius_x_right + 1));
  Mat sub_orient = orient(Range(radius_y_up, radius_y_down + 1),
                          Range(radius_x_left, radius_x_right + 1));

  // Mat W = sub_amplit.mul(gauss_weight);
  // //加入高斯权重，计算高斯权重时，正确匹配点对反而变少了
  Mat W = sub_amplit; //没加高斯权重，梯度幅值

  //计算直方图
  AutoBuffer<float> buffer(n + 4);

  float *temp_hist = buffer + 2;

  for (int i = 0; i < n; ++i)
    temp_hist[i] = 0.f;

  for (int i = 0; i < sub_orient.rows; i++) {
    float *ptr_1 = W.ptr<float>(i);
    float *ptr_2 = sub_orient.ptr<float>(i);

    for (int j = 0; j < sub_orient.cols; j++) {
      if (((i - center_y) * (i - center_y) + (j - center_x) * (j - center_x)) <
          radius * radius) {
        int bin = cvRound(ptr_2[j] * n / 360.f);

        if (bin > n)
          bin = bin - n;
        if (bin < 0)
          bin = bin + n;
        temp_hist[bin] += ptr_1[j];
      }
    }
  }

  //平滑直方图，可以防止突变
  temp_hist[-1] = temp_hist[n - 1];
  temp_hist[-2] = temp_hist[n - 2];
  temp_hist[n] = temp_hist[0];
  temp_hist[n + 1] = temp_hist[1];

  for (int i = 0; i < n; ++i) {
    hist[i] = (temp_hist[i - 2] + temp_hist[i + 2]) * (1.f / 16.f) +
              (temp_hist[i - 1] + temp_hist[i + 1]) * (4.f / 16.f) +
              temp_hist[i] * (6.f / 16.f);
  }

  //获得直方图中最大值
  float max_value = hist[0];
  for (int i = 1; i < n; ++i) {
    if (hist[i] > max_value)
      max_value = hist[i];
  }
  return max_value;
}

/****************************该函数精确定位特征点位置(x,y,scale)，用于后面特征点的检测*************************/
/*功能：确定特征点的位置，并通过主曲率消除边缘相应点,该版本是简化版
 dog_pry表示DOG金字塔
 kpt表示精确定位后该特征点的信息
 octave表示初始特征点所在的组
 layer表示初始特征点所在的层
 row表示初始特征点在图像中的行坐标
 col表示初始特征点在图像中的列坐标
 nOctaveLayers表示DOG金字塔每组中间层数，默认是3
 contrastThreshold表示对比度阈值，默认是0.04
 edgeThreshold表示边缘阈值，默认是10
 sigma表示高斯尺度空间最底层图像尺度，默认是1.6*/
static bool adjust_local_extrema_1(const vector<vector<Mat>> &dog_pyr,
                                   KeyPoint &kpt, int octave, int &layer,
                                   int &row, int &col, int nOctaveLayers,
                                   float contrastThreshold, float edgeThreshold,
                                   float sigma) {
  float xi = 0, xr = 0, xc = 0;
  int i = 0;
  for (; i < MAX_INTERP_STEPS; ++i) //最大迭代次数
  {
    const Mat &img = dog_pyr[octave][layer];      //当前层图像索引
    const Mat &prev = dog_pyr[octave][layer - 1]; //之前层图像索引
    const Mat &next = dog_pyr[octave][layer + 1]; //下一层图像索引

    //特征点位置x方向，y方向,尺度方向的一阶偏导数
    float dx = (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) *
               (1.f / 2.f);
    float dy = (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) *
               (1.f / 2.f);
    float dz =
        (next.at<float>(row, col) - prev.at<float>(row, col)) * (1.f / 2.f);

    //计算特征点位置二阶偏导数
    float v2 = img.at<float>(row, col);
    float dxx =
        img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - 2 * v2;
    float dyy =
        img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - 2 * v2;
    float dzz = prev.at<float>(row, col) + next.at<float>(row, col) - 2 * v2;

    //计算特征点周围混合二阶偏导数
    float dxy =
        (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
         img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) *
        (1.f / 4.f);
    float dxz = (next.at<float>(row, col + 1) + prev.at<float>(row, col - 1) -
                 next.at<float>(row, col - 1) - prev.at<float>(row, col + 1)) *
                (1.f / 4.f);
    float dyz = (next.at<float>(row + 1, col) + prev.at<float>(row - 1, col) -
                 next.at<float>(row - 1, col) - prev.at<float>(row + 1, col)) *
                (1.f / 4.f);

    Matx33f H(dxx, dxy, dxz, dxy, dyy, dyz, dxz, dyz, dzz);

    Vec3f dD(dx, dy, dz);

    Vec3f X = H.solve(dD, DECOMP_SVD);

    xc = -X[0]; // x方向偏移量
    xr = -X[1]; // y方向偏移量
    xi = -X[2]; //尺度方向偏移量

    //如果三个方向偏移量都小于0.5，说明已经找到特征点准确位置
    if (abs(xc) < 0.5f && abs(xr) < 0.5f && abs(xi) < 0.5f)
      break;

    //如果其中一个方向的偏移量过大，则删除该点
    if (abs(xc) > (float)(INT_MAX / 3) || abs(xr) > (float)(INT_MAX / 3) ||
        abs(xi) > (float)(INT_MAX / 3))
      return false;

    col = col + cvRound(xc);
    row = row + cvRound(xr);
    layer = layer + cvRound(xi);

    //如果特征点定位在边界区域，同样也需要删除
    if (layer < 1 || layer > nOctaveLayers || col < IMG_BORDER ||
        col > img.cols - IMG_BORDER || row < IMG_BORDER ||
        row > img.rows - IMG_BORDER)
      return false;
  }

  //如果i=MAX_INTERP_STEPS，说明循环结束也没有满足条件，即该特征点需要被删除
  if (i >= MAX_INTERP_STEPS)
    return false;

  /**************************再次删除低响应点(对比度较低的点)********************************/
  //再次计算特征点位置x方向，y方向,尺度方向的一阶偏导数
  //高对比度的特征对图像的变形是稳定的
  {
    const Mat &img = dog_pyr[octave][layer];
    const Mat &prev = dog_pyr[octave][layer - 1];
    const Mat &next = dog_pyr[octave][layer + 1];

    float dx = (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) *
               (1.f / 2.f);
    float dy = (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) *
               (1.f / 2.f);
    float dz =
        (next.at<float>(row, col) - prev.at<float>(row, col)) * (1.f / 2.f);
    Matx31f dD(dx, dy, dz);
    float t = dD.dot(Matx31f(xc, xr, xi));

    float contr =
        img.at<float>(row, col) + t * 0.5f; //特征点响应 |D(x~)| 即对比度

    // Low建议contr阈值是0.03，但是RobHess等建议阈值为0.04/nOctaveLayers
    if (abs(contr) <
        contrastThreshold / nOctaveLayers) //阈值设为0.03时特征点数量过多
      return false;

    /******************************删除边缘响应比较强的点************************************/

    //再次计算特征点位置二阶偏导数，获取特征点出的 Hessian 矩阵，主曲率通过 2X2
    //的 Hessian 矩阵求出
    //一个定义不好的高斯差分算子的极值在横跨边缘的地方有较大的主曲率而在垂直边缘的方向有较小的主曲率
    float v2 = img.at<float>(row, col);
    float dxx =
        img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - 2 * v2;
    float dyy =
        img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - 2 * v2;
    float dxy =
        (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
         img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) *
        (1.f / 4.f);
    float det = dxx * dyy - dxy * dxy;
    float trace = dxx + dyy;

    //主曲率和阈值的对比判定
    if (det < 0 || (trace * trace * edgeThreshold >=
                    det * (edgeThreshold + 1) * (edgeThreshold + 1)))
      return false;

    /*********到目前为止该特征的满足上面所有要求，保存该特征点信息***********/
    kpt.pt.x = ((float)col + xc) * (1 << octave); //相对于最底层的图像的x坐标
    kpt.pt.y = ((float)row + xr) * (1 << octave); //相对于最底层图像的y坐标
    kpt.octave = octave + (layer << 8); //组号保存在低字节，层号保存在高字节
    //相对于最底层图像的尺度
    kpt.size = sigma * powf(2.f, (layer + xi) / nOctaveLayers) * (1 << octave);
    kpt.response = abs(contr); //特征点响应值(对比度)

    return true;
  }
}

/****************************该函数精确定位特征点位置(x,y,scale)，用于后面特征点的检测*************************/
//该版本是 SIFT 原版，检测得到的特征点数量更多
static bool adjust_local_extrema_2(const vector<vector<Mat>> &dog_pyr,
                                   KeyPoint &kpt, int octave, int &layer,
                                   int &row, int &col, int nOctaveLayers,
                                   float contrastThreshold, float edgeThreshold,
                                   float sigma) {
  const float img_scale = 1.f / (255 * SIFT_FIXPT_SCALE); // SIFT_FIXPT_SCALE=48
  const float deriv_scale = img_scale * 0.5f;
  const float second_deriv_scale = img_scale;
  const float cross_deriv_scale = img_scale * 0.25f;

  float xi = 0, xr = 0, xc = 0;
  int i = 0;

  for (; i < MAX_INTERP_STEPS; ++i) //最大迭代次数
  {
    const Mat &img = dog_pyr[octave][layer];      //当前层图像索引
    const Mat &prev = dog_pyr[octave][layer - 1]; //之前层图像索引
    const Mat &next = dog_pyr[octave][layer + 1]; //下一层图像索引

    //计算一阶偏导数，通过临近点差分求得
    float dx = (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) *
               deriv_scale;
    float dy = (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) *
               deriv_scale;
    float dz =
        (next.at<float>(row, col) - prev.at<float>(row, col)) * deriv_scale;

    //计算特征点位置二阶偏导数
    // float v2  = img.at<float>(row, col);
    float v2 = (float)img.at<float>(row, col) * 2.f;
    float dxx =
        (img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - v2) *
        second_deriv_scale;
    float dyy =
        (img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - v2) *
        second_deriv_scale;
    float dzz = (prev.at<float>(row, col) + next.at<float>(row, col) - v2) *
                second_deriv_scale;

    //计算特征点周围混合二阶偏导数
    float dxy =
        (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
         img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) *
        cross_deriv_scale;
    float dxz = (next.at<float>(row, col + 1) + prev.at<float>(row, col - 1) -
                 next.at<float>(row, col - 1) - prev.at<float>(row, col + 1)) *
                cross_deriv_scale;
    float dyz = (next.at<float>(row + 1, col) + prev.at<float>(row - 1, col) -
                 next.at<float>(row - 1, col) - prev.at<float>(row + 1, col)) *
                cross_deriv_scale;

    Matx33f H(dxx, dxy, dxz, dxy, dyy, dyz, dxz, dyz, dzz);

    Vec3f dD(dx, dy, dz);

    Vec3f X = H.solve(dD, DECOMP_SVD);

    xc = -X[0]; // x方向偏移量
    xr = -X[1]; // y方向偏移量
    xi = -X[2]; //尺度方向偏移量

    //如果三个方向偏移量都小于0.5，说明已经找到特征点准确位置
    if (abs(xc) < 0.5f && abs(xr) < 0.5f && abs(xi) < 0.5f)
      break;

    //如果其中一个方向的偏移量过大，则删除该点
    if (abs(xc) > (float)(INT_MAX / 3) || abs(xr) > (float)(INT_MAX / 3) ||
        abs(xi) > (float)(INT_MAX / 3))
      return false;

    col = col + cvRound(xc);
    row = row + cvRound(xr);
    layer = layer + cvRound(xi);

    //如果特征点定位在边界区域，同样也需要删除
    if (layer < 1 || layer > nOctaveLayers || col < IMG_BORDER ||
        col >= img.cols - IMG_BORDER || row < IMG_BORDER ||
        row >= img.rows - IMG_BORDER)
      return false;
  }
  //如果i=MAX_INTERP_STEPS，说明循环结束也没有满足条件，即该特征点需要被删除
  if (i >= MAX_INTERP_STEPS)
    return false;

  /**************************再次删除低响应点(对比度较低的点)********************************/
  //再次计算特征点位置x方向，y方向,尺度方向的一阶偏导数
  //高对比度的特征对图像的变形是稳定的

  const Mat &img = dog_pyr[octave][layer];
  const Mat &prev = dog_pyr[octave][layer - 1];
  const Mat &next = dog_pyr[octave][layer + 1];

  float dx =
      (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) * deriv_scale;
  float dy =
      (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) * deriv_scale;
  float dz =
      (next.at<float>(row, col) - prev.at<float>(row, col)) * deriv_scale;
  Matx31f dD(dx, dy, dz);
  float t = dD.dot(Matx31f(xc, xr, xi));

  float contr =
      img.at<float>(row, col) + t * 0.5f; //特征点响应 |D(x~)| 即对比度

  // Low建议contr阈值是0.03，但是RobHess等建议阈值为0.04/nOctaveLayers
  if (abs(contr) <
      contrastThreshold / nOctaveLayers) //阈值设为0.03时特征点数量过多
    return false;

  /******************************删除边缘响应比较强的点************************************/

  //再次计算特征点位置二阶偏导数，获取特征点出的 Hessian 矩阵，主曲率通过 2X2 的
  // Hessian 矩阵求出
  //一个定义不好的高斯差分算子的极值在横跨边缘的地方有较大的主曲率而在垂直边缘的方向有较小的主曲率
  float v2 = (float)img.at<float>(row, col) * 2.f;
  float dxx = (img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - v2) *
              second_deriv_scale;
  float dyy = (img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - v2) *
              second_deriv_scale;
  float dxy =
      (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
       img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) *
      cross_deriv_scale;

  float det = dxx * dyy - dxy * dxy;
  float trace = dxx + dyy;

  //主曲率和阈值的对比判定
  if (det <= 0 || (trace * trace * edgeThreshold >=
                   det * (edgeThreshold + 1) * (edgeThreshold + 1)))
    return false;

  /*********保存该特征点信息***********/
  kpt.pt.x =
      ((float)col + xc) * (1 << octave); //高斯金字塔坐标根据组数扩大相应的倍数
  kpt.pt.y = ((float)row + xr) * (1 << octave);

  // SIFT 描述子
  kpt.octave =
      octave + (layer << 8) +
      (cvRound((xi + 0.5) * 255) << 16); //特征点被检测出时所在的金字塔组

  kpt.size = sigma * powf(2.f, (layer + xi) / nOctaveLayers) * (1 << octave) *
             2;              //关键点邻域直径
  kpt.response = abs(contr); //特征点响应值(对比度)

  return true;
}

/************该函数在DOG金字塔上进行特征点检测，特征点精确定位，删除低对比度点，删除边缘响应较大点**********/
/*dog_pyr表示高斯金字塔			原始 SIFT 算子
 gauss_pyr表示高斯金字塔
 keypoints表示检测到的特征点*/
void mySift::find_scale_space_extrema(const vector<vector<Mat>> &dog_pyr,
                                      const vector<vector<Mat>> &gauss_pyr,
                                      vector<KeyPoint> &keypoints) {
  int nOctaves = (int)dog_pyr.size(); //子八度个数

  // Low文章建议thresholdThreshold是0.03，Rob
  // Hess等人使用0.04/nOctaveLayers作为阈值
  float threshold = (float)(contrastThreshold / nOctaveLayers);
  const int n = ORI_HIST_BINS; // n=36
  float hist[n];
  KeyPoint kpt;

  keypoints.clear(); //先清空keypoints

  for (int i = 0; i < nOctaves; ++i) //对于每一组
  {
    int numKeys = 0;
    for (int j = 1; j <= nOctaveLayers; ++j) //对于组内每一层
    {
      const Mat &curr_img = dog_pyr[i][j];     //当前层
      const Mat &prev_img = dog_pyr[i][j - 1]; //上一层
      const Mat &next_img = dog_pyr[i][j + 1]; //下一层

      int num_row = curr_img.rows;
      int num_col = curr_img.cols;    //获得当前组图像的大小
      size_t step = curr_img.step1(); //一行元素所占字节数

      //遍历每一个尺度层中的有效像素，像素值
      for (int r = IMG_BORDER; r < num_row - IMG_BORDER; ++r) {
        const float *curr_ptr = curr_img.ptr<float>(
            r); //指向的是第 r 行的起点，返回的是 float 类型的像素值
        const float *prev_ptr = prev_img.ptr<float>(r - 1);
        const float *next_ptr = next_img.ptr<float>(r + 1);

        for (int c = IMG_BORDER; c < num_col - IMG_BORDER; ++c) {
          float val = curr_ptr[c]; //当前中心点响应值

          //开始检测特征点
          if (abs(val) > threshold &&
              ((val > 0 && val >= curr_ptr[c - 1] && val >= curr_ptr[c + 1] &&
                val >= curr_ptr[c - step - 1] && val >= curr_ptr[c - step] &&
                val >= curr_ptr[c - step + 1] &&
                val >= curr_ptr[c + step - 1] && val >= curr_ptr[c + step] &&
                val >= curr_ptr[c + step + 1] && val >= prev_ptr[c] &&
                val >= prev_ptr[c - 1] && val >= prev_ptr[c + 1] &&
                val >= prev_ptr[c - step - 1] && val >= prev_ptr[c - step] &&
                val >= prev_ptr[c - step + 1] &&
                val >= prev_ptr[c + step - 1] && val >= prev_ptr[c + step] &&
                val >= prev_ptr[c + step + 1] && val >= next_ptr[c] &&
                val >= next_ptr[c - 1] && val >= next_ptr[c + 1] &&
                val >= next_ptr[c - step - 1] && val >= next_ptr[c - step] &&
                val >= next_ptr[c - step + 1] &&
                val >= next_ptr[c + step - 1] && val >= next_ptr[c + step] &&
                val >= next_ptr[c + step + 1]) ||
               (val < 0 && val <= curr_ptr[c - 1] && val <= curr_ptr[c + 1] &&
                val <= curr_ptr[c - step - 1] && val <= curr_ptr[c - step] &&
                val <= curr_ptr[c - step + 1] &&
                val <= curr_ptr[c + step - 1] && val <= curr_ptr[c + step] &&
                val <= curr_ptr[c + step + 1] && val <= prev_ptr[c] &&
                val <= prev_ptr[c - 1] && val <= prev_ptr[c + 1] &&
                val <= prev_ptr[c - step - 1] && val <= prev_ptr[c - step] &&
                val <= prev_ptr[c - step + 1] &&
                val <= prev_ptr[c + step - 1] && val <= prev_ptr[c + step] &&
                val <= prev_ptr[c + step + 1] && val <= next_ptr[c] &&
                val <= next_ptr[c - 1] && val <= next_ptr[c + 1] &&
                val <= next_ptr[c - step - 1] && val <= next_ptr[c - step] &&
                val <= next_ptr[c - step + 1] &&
                val <= next_ptr[c + step - 1] && val <= next_ptr[c + step] &&
                val <= next_ptr[c + step + 1]))) {
            ++numKeys;
            //获得特征点初始行号，列号，组号，组内层号
            int octave = i, layer = j, r1 = r, c1 = c;

            if (!adjust_local_extrema_1(dog_pyr, kpt, octave, layer, r1, c1,
                                        nOctaveLayers, (float)contrastThreshold,
                                        (float)edgeThreshold, (float)sigma)) {
              continue; //如果该初始点不满足条件，则不保存改点
            }

            float scale =
                kpt.size / float(1 << octave); //该特征点相对于本组的尺度

            // max_hist值对应的方向为主方向
            float max_hist = clac_orientation_hist(
                gauss_pyr[octave][layer], Point(c1, r1), scale, n, hist);

            //大于mag_thr值对应的方向为辅助方向
            float mag_thr =
                max_hist * ORI_PEAK_RATIO; //主峰值 80% 的方向作为辅助方向

            //遍历直方图中的 36 个bin
            for (int i = 0; i < n; ++i) {
              int left = i > 0 ? i - 1 : n - 1;
              int right = i < n - 1 ? i + 1 : 0;

              //创建新的特征点，大于主峰值 80%
              //的方向，赋值给该特征点，作为一个新的特征点；即有多个特征点，位置、尺度相同，方向不同
              if (hist[i] > hist[left] && hist[i] > hist[right] &&
                  hist[i] >= mag_thr) {
                float bin = i + 0.5f * (hist[left] - hist[right]) /
                                    (hist[left] + hist[right] - 2 * hist[i]);
                bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;

                kpt.angle = (360.f / n) *
                            bin; //原始 SIFT 算子使用的特征点的主方向0-360度
                keypoints.push_back(kpt); //保存该特征点
              }
            }
          }
        }
      }
    }
    LOG(INFO,"numKeys : %d", numKeys);
  }

  // cout << "初始满足要求特征点个数是: " << numKeys << endl;
}

/************该函数在DOG金字塔上进行特征点检测，特征点精确定位，删除低对比度点，删除边缘响应较大点**********/
//对特征点进行方向的细化 + 增加更多的主方向版本
//——此时细化是对最后要给关键点进行赋值时的细化
//还可以考虑直接对方向直方图进行细化
void mySift::find_scale_space_extrema1(const vector<vector<Mat>> &dog_pyr,
                                       vector<vector<Mat>> &gauss_pyr,
                                       vector<KeyPoint> &keypoints) {
  int nOctaves = (int)dog_pyr.size(); //子八度个数

  // Low文章建议thresholdThreshold是0.03，Rob
  // Hess等人使用0.04/nOctaveLayers作为阈值
  float threshold = (float)(contrastThreshold / nOctaveLayers);
  const int n = ORI_HIST_BINS; // n=36
  float hist[n];
  KeyPoint kpt;

  vector<Mat> amplit; //存放高斯差分金字塔每一层的梯度幅度图像
  vector<Mat> orient; //存放高斯差分金字塔每一层的梯度方向图像

  keypoints.clear(); //先清空keypoints

  for (int i = 0; i < nOctaves; ++i) //对于每一组
  {
    int numKeys = 0;
    for (int j = 1; j <= nOctaveLayers; ++j) //对于组内每一层
    {
      const Mat &curr_img = dog_pyr[i][j];     //当前层
      const Mat &prev_img = dog_pyr[i][j - 1]; //上一层
      const Mat &next_img =
          dog_pyr[i][j + 1]; //下一层
                             // ----------------------------------------------
                             // std::cout << curr_img << std::endl;
                             // Mat curr_img_copy = curr_img.clone();
                             // ----------------------------------------------
      int num_row = curr_img.rows;
      int num_col = curr_img.cols;    //获得当前组图像的大小
      size_t step = curr_img.step1(); //一行元素所占字节数

      //遍历每一个尺度层中的有效像素，像素值
      for (int r = IMG_BORDER; r < num_row - IMG_BORDER; ++r) {
        const float *curr_ptr = curr_img.ptr<float>(
            r); //指向的是第 r 行的起点，返回的是 float 类型的像素值
        const float *prev_ptr = prev_img.ptr<float>(r);
        const float *next_ptr = next_img.ptr<float>(r);

        for (int c = IMG_BORDER; c < num_col - IMG_BORDER; ++c) {
          float val = curr_ptr[c]; //当前中心点响应值

          //开始检测特征点
          if (abs(val) > threshold &&
              ((val > 0 && val >= curr_ptr[c - 1] && val >= curr_ptr[c + 1] &&
                val >= curr_ptr[c - step - 1] && val >= curr_ptr[c - step] &&
                val >= curr_ptr[c - step + 1] &&
                val >= curr_ptr[c + step - 1] && val >= curr_ptr[c + step] &&
                val >= curr_ptr[c + step + 1] && val >= prev_ptr[c] &&
                val >= prev_ptr[c - 1] && val >= prev_ptr[c + 1] &&
                val >= prev_ptr[c - step - 1] && val >= prev_ptr[c - step] &&
                val >= prev_ptr[c - step + 1] &&
                val >= prev_ptr[c + step - 1] && val >= prev_ptr[c + step] &&
                val >= prev_ptr[c + step + 1] && val >= next_ptr[c] &&
                val >= next_ptr[c - 1] && val >= next_ptr[c + 1] &&
                val >= next_ptr[c - step - 1] && val >= next_ptr[c - step] &&
                val >= next_ptr[c - step + 1] &&
                val >= next_ptr[c + step - 1] && val >= next_ptr[c + step] &&
                val >= next_ptr[c + step + 1]) ||
               (val < 0 && val <= curr_ptr[c - 1] && val <= curr_ptr[c + 1] &&
                val <= curr_ptr[c - step - 1] && val <= curr_ptr[c - step] &&
                val <= curr_ptr[c - step + 1] &&
                val <= curr_ptr[c + step - 1] && val <= curr_ptr[c + step] &&
                val <= curr_ptr[c + step + 1] && val <= prev_ptr[c] &&
                val <= prev_ptr[c - 1] && val <= prev_ptr[c + 1] &&
                val <= prev_ptr[c - step - 1] && val <= prev_ptr[c - step] &&
                val <= prev_ptr[c - step + 1] &&
                val <= prev_ptr[c + step - 1] && val <= prev_ptr[c + step] &&
                val <= prev_ptr[c + step + 1] && val <= next_ptr[c] &&
                val <= next_ptr[c - 1] && val <= next_ptr[c + 1] &&
                val <= next_ptr[c - step - 1] && val <= next_ptr[c - step] &&
                val <= next_ptr[c - step + 1] &&
                val <= next_ptr[c + step - 1] && val <= next_ptr[c + step] &&
                val <= next_ptr[c + step + 1]))) {
            ++numKeys;
            //获得特征点初始行号，列号，组号，组内层号
            int octave = i, layer = j, r1 = r, c1 = c, nums = i * nOctaves + j;
            // cv::circle(curr_img_copy, cv::Point(r, c), 3, cv::Scalar(0, 0,
            // 255), 1);
            if (!adjust_local_extrema_2(dog_pyr, kpt, octave, layer, r1, c1,
                                        nOctaveLayers, (float)contrastThreshold,
                                        (float)edgeThreshold, (float)sigma)) {
              continue; //如果该初始点不满足条件，则不保存改点
            }

            float scale =
                kpt.size / float(1 << octave); //该特征点相对于本组的尺度

            //计算梯度幅度和梯度方向
            // amplit_orient(curr_img, amplit, orient, scale, nums);

            // max_hist值对应的方向为主方向
            float max_hist = clac_orientation_hist(
                gauss_pyr[octave][layer], Point(c1, r1), scale, n, hist);
            // float max_hist = calc_orient_hist(amplit[nums], orient[nums],
            // Point2f(c1, r1), scale, hist, n);

            //大于mag_thr值对应的方向为辅助方向
            // float mag_thr = max_hist * ORI_PEAK_RATIO;	//主峰值 80%
            // 的方向作为辅助方向

            //增加更多的主方向，以增加特征点对梯度差异的鲁棒性
            float sum = 0.0;     //直方图对应的幅值之和
            float mag_thr = 0.0; //判断是否为主方向的阈值

            for (int i = 0; i < n; ++i) {
              sum += hist[i];
            }
            mag_thr = 0.5 * (1.0 / 36) * sum;

            //遍历直方图中的 36 个bin
            for (int i = 0; i < n; ++i) {
              int left = i > 0 ? i - 1 : n - 1;
              int right = i < n - 1 ? i + 1 : 0;

              //创建新的特征点，大于主峰值 80%
              //的方向，赋值给该特征点，作为一个新的特征点；即有多个特征点，位置、尺度相同，方向不同
              if (hist[i] > hist[left] && hist[i] > hist[right] &&
                  hist[i] >= mag_thr) {
                float bin = i + 0.5f * (hist[left] - hist[right]) /
                                    (hist[left] + hist[right] - 2 * hist[i]);
                bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;

                //修改的地方，特征点的主方向修改为了0-180度，相当于对方向做了一个细化
                float angle = (360.f / n) * bin;
                if (angle >= 1 && angle <= 180) {
                  kpt.angle = angle;
                } else if (angle > 180 && angle < 360) {
                  kpt.angle = 360 - angle;
                }

                // kpt.angle = (360.f / n) * bin;			//原始
                // SIFT 算子使用的特征点的主方向0-360度
                // LOG(INFO,"[%d %d %d]",i,left,right);
                // LOG(INFO,"point[%f,%f] angle: %f", kpt.pt.x, kpt.pt.y, kpt.angle);
                keypoints.push_back(kpt); //保存该特征点
              }
            }
          }
        }
      }
      // cv::imshow("curr_img_copy", curr_img_copy);
      // cv::waitKey(0);
    }
   LOG(INFO, "numKeys: %d", numKeys);
   LOG(INFO, "keypoints.size(): %ld", keypoints.size());
  }
  // cout << "初始满足要求特征点个数是: " << numKeys << endl;
}

/*该函数生成matlab中的meshgrid函数*/
/*x_range表示x方向的范围
y_range表示y方向的范围
X表示生成的沿x轴变化的网格
Y表示生成沿y轴变换的网格
*/
static void meshgrid(const Range &x_range, const Range &y_range, Mat &X,
                     Mat &Y) {
  int x_start = x_range.start, x_end = x_range.end;
  int y_start = y_range.start, y_end = y_range.end;
  int width = x_end - x_start + 1;
  int height = y_end - y_start + 1;

  X.create(height, width, CV_32FC1);
  Y.create(height, width, CV_32FC1);

  for (int i = y_start; i <= y_end; i++) {
    float *ptr_1 = X.ptr<float>(i - y_start);
    for (int j = x_start; j <= x_end; ++j)
      ptr_1[j - x_start] = j * 1.0f;
  }

  for (int i = y_start; i <= y_end; i++) {
    float *ptr_2 = Y.ptr<float>(i - y_start);
    for (int j = x_start; j <= x_end; ++j)
      ptr_2[j - x_start] = i * 1.0f;
  }
}

/******************************计算一个特征点的描述子***********************************/
/*gauss_image表示特征点所在的高斯层
  main_angle表示特征点的主方向，角度范围是0-360度
  pt表示特征点在高斯图像上的坐标，相对与本组，不是相对于最底层
  scale表示特征点所在层的尺度，相对于本组，不是相对于最底层
  d表示特征点邻域网格宽度
  n表示每个网格内像素梯度角度等分个数
  descriptor表示生成的特征点的描述子*/
static void calc_sift_descriptor(const Mat &gauss_image, float main_angle,
                                 Point2f pt, float scale, int d, int n,
                                 float *descriptor) {
  Point ptxy(cvRound(pt.x), cvRound(pt.y)); //坐标取整

  float cos_t = cosf(-main_angle *
                     (float)(CV_PI / 180)); //把角度转化为弧度，计算主方向的余弦
  float sin_t = sinf(-main_angle *
                     (float)(CV_PI / 180)); //把角度转化为弧度，计算主方向的正弦

  float bins_per_rad = n / 360.f; // n = 8 ，梯度直方图分为 8 个方向

  float exp_scale = -1.f / (d * d * 0.5f); //权重指数部分

  float hist_width =
      DESCR_SCL_FCTR * scale; //特征点邻域内子区域边长，子区域的边长

  int radius = cvRound(hist_width * (d + 1) * sqrt(2) *
                       0.5f); //特征点邻域半径(d+1)*(d+1)，四舍五入

  int rows = gauss_image.rows, cols = gauss_image.cols; //当前高斯层行、列信息

  //特征点邻域半径
  radius = min(radius, (int)sqrt((double)rows * rows + cols * cols));

  cos_t = cos_t / hist_width;
  sin_t = sin_t / hist_width;

  int len = (2 * radius + 1) *
            (2 * radius + 1); //邻域内总像素数，为后面动态分配内存使用

  int histlen = (d + 2) * (d + 2) * (n + 2); //值为 360

  AutoBuffer<float> buf(6 * len + histlen);

  // X保存水平差分，Y保存竖直差分，Mag保存梯度幅度，Angle保存特征点方向,
  // W保存高斯权重
  float *X = buf, *Y = buf + len, *Mag = Y, *Angle = Y + len, *W = Angle + len;
  float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

  //首先清空直方图数据
  for (int i = 0; i < d + 2; ++i) // i 对应 row
  {
    for (int j = 0; j < d + 2; ++j) // j 对应 col
    {
      for (int k = 0; k < n + 2; ++k)

        hist[(i * (d + 2) + j) * (n + 2) + k] = 0.f;
    }
  }

  //把邻域内的像素分配到相应子区域内，计算子区域内每个像素点的权重(子区域即 d*d
  //中每一个小方格)
  int k = 0;

  //实际上是在 4 x 4 的网格中找 16 个种子点，每个种子点都在子网格的正中心，
  //通过三线性插值对不同种子点间的像素点进行加权作用到不同的种子点上绘制方向直方图

  for (int i = -radius; i < radius; ++i) // i 对应 y 行坐标
  {
    for (int j = -radius; j < radius; ++j) // j 对应 x 列坐标
    {
      float c_rot = j * cos_t - i * sin_t; //旋转后邻域内采样点的 x 坐标
      float r_rot = j * sin_t + i * cos_t; //旋转后邻域内采样点的 y 坐标

      //旋转后 5 x 5 的网格中的所有像素点被分配到 4 x 4 的网格中
      float cbin = c_rot + d / 2 - 0.5f; //旋转后的采样点落在子区域的 x 坐标
      float rbin = r_rot + d / 2 - 0.5f; //旋转后的采样点落在子区域的 y 坐标

      int r = ptxy.y + i, c = ptxy.x + j; // ptxy是高斯金字塔中的坐标

      //这里rbin，cbin范围是(-1,d)
      if (rbin > -1 && rbin < d && cbin > -1 && cbin < d && r > 0 &&
          r < rows - 1 && c > 0 && c < cols - 1) {
        float dx =
            gauss_image.at<float>(r, c + 1) - gauss_image.at<float>(r, c - 1);
        float dy =
            gauss_image.at<float>(r + 1, c) - gauss_image.at<float>(r - 1, c);

        X[k] = dx; //邻域内所有像素点的水平差分
        Y[k] = dy; //邻域内所有像素点的竖直差分

        CBin[k] = cbin; //邻域内所有采样点落在子区域的 x 坐标
        RBin[k] = rbin; //邻域内所有采样点落在子区域的 y 坐标

        W[k] = (c_rot * c_rot + r_rot * r_rot) * exp_scale; //高斯权值的指数部分

        ++k;
      }
    }
  }

  //计算采样点落在子区域的像素梯度幅度，梯度角度，和高斯权值
  len = k;

  cv::hal::exp(W, W, len); //邻域内所有采样点落在子区域的像素的高斯权值
  cv::hal::fastAtan2(
      Y, X, Angle, len,
      true); //邻域内所有采样点落在子区域的像素的梯度方向，角度范围是0-360度
  cv::hal::magnitude(X, Y, Mag,
                     len); //邻域内所有采样点落在子区域的像素的梯度幅度

  //实际上是在 4 x 4 的网格中找 16 个种子点，每个种子点都在子网格的正中心，
  //通过三线性插值对不同种子点间的像素点进行加权作用到不同的种子点上绘制方向直方图

  //计算每个特征点的特征描述子
  for (k = 0; k < len; ++k) {
    float rbin = RBin[k],
          cbin = CBin[k]; //子区域内像素点坐标，rbin,cbin范围是(-1,d)

    // 改进的地方，对方向进行了一个细化，也是为了增加对梯度差异的鲁棒性
    // if (Angle[k] > 180 && Angle[k] < 360)
    //	Angle[k] = 360 - Angle[k];

    //子区域内像素点处理后的方向
    float temp = Angle[k] - main_angle;

    /*if (temp > 180 && temp < 360)
            temp = 360 - temp;*/

    float obin = temp * bins_per_rad; //指定方向的数量后，邻域内像素点对应的方向

    float mag = Mag[k] * W[k]; //子区域内像素点乘以权值后的梯度幅值

    int r0 = cvFloor(rbin); // ro取值集合是{-1，0，1，2，3}，没太懂为什么？
    int c0 = cvFloor(cbin); // c0取值集合是{-1，0，1，2，3}
    int o0 = cvFloor(obin);

    rbin = rbin -
           r0; //子区域内像素点坐标的小数部分，用于线性插值，分配像素点的作用
    cbin = cbin - c0;
    obin = obin - o0; //子区域方向的小数部分

    //限制范围为梯度直方图横坐标[0,n)，8 个方向直方图
    if (o0 < 0)
      o0 = o0 + n;
    if (o0 >= n)
      o0 = o0 - n;

    //三线性插值用于计算落在两个子区域之间的像素对两个子区域的作用，并把其分配到对应子区域的8个方向上
    //像素对应的信息通过加权分配给其周围的种子点，并把相应方向的梯度值进行累加

    float v_r1 = mag * rbin; //第二行分配的值
    float v_r0 = mag - v_r1; //第一行分配的值

    float v_rc11 = v_r1 * cbin; //第二行第二列分配的值，右下角种子点
    float v_rc10 = v_r1 - v_rc11; //第二行第一列分配的值，左下角种子点

    float v_rc01 = v_r0 * cbin; //第一行第二列分配的值，右上角种子点
    float v_rc00 = v_r0 - v_rc01; //第一行第一列分配的值，左上角种子点

    //一个像素点的方向为每个种子点的两个方向做出贡献
    float v_rco111 = v_rc11 * obin; //右下角种子点第二个方向上分配的值
    float v_rco110 = v_rc11 - v_rco111; //右下角种子点第一个方向上分配的值

    float v_rco101 = v_rc10 * obin;
    float v_rco100 = v_rc10 - v_rco101;

    float v_rco011 = v_rc01 * obin;
    float v_rco010 = v_rc01 - v_rco011;

    float v_rco001 = v_rc00 * obin;
    float v_rco000 = v_rc00 - v_rco001;

    //该像素所在网格的索引
    int idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
    // LOG(DEBUG,"idx : %d,n : %d,d : %d",idx,n,d);
    hist[idx] += v_rco000;
    hist[idx + 1] += v_rco001;
    hist[idx + n + 2] += v_rco010;
    hist[idx + n + 3] += v_rco011;
    hist[idx + (d + 2) * (n + 2)] += v_rco100;
    hist[idx + (d + 2) * (n + 2) + 1] += v_rco101;
    hist[idx + (d + 3) * (n + 2)] += v_rco110;
    hist[idx + (d + 3) * (n + 2) + 1] += v_rco111;
  }

  //由于圆周循环的特性，对计算以后幅角小于 0 度或大于 360 度的值重新进行调整，使
  //其在 0～360 度之间
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < d; ++j) {
      //类似于 hist[0][2][3] 第 0 行，第 2 列，种子点直方图中的第 3 个 bin
      int idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);

      hist[idx] += hist[idx + n];
      // hist[idx + 1] += hist[idx + n +
      // 1];//opencv源码中这句话是多余的,hist[idx + n + 1]永远是0.0

      for (k = 0; k < n; ++k)
        descriptor[(i * d + j) * n + k] = hist[idx + k];
    }
  }

  //对描述子进行归一化
  int lenght = d * d * n;
  float norm = 0;

  //计算特征描述向量的模值的平方
  for (int i = 0; i < lenght; ++i) {
    norm = norm + descriptor[i] * descriptor[i];
  }
  norm = sqrt(norm); //特征描述向量的模值

  //此次归一化能去除光照的影响
  for (int i = 0; i < lenght; ++i) {
    descriptor[i] = descriptor[i] / norm;
  }

  //阈值截断,去除特征描述向量中大于 0.2
  //的值，能消除非线性光照的影响(相机饱和度对某些放的梯度影响较大，对方向的影响较小)
  for (int i = 0; i < lenght; ++i) {
    descriptor[i] = min(descriptor[i], DESCR_MAG_THR);
  }

  //再次归一化，能够提高特征的独特性
  norm = 0;
  for (int i = 0; i < lenght; ++i) {
    norm = norm + descriptor[i] * descriptor[i];
  }
  norm = sqrt(norm);
  for (int i = 0; i < lenght; ++i) {
    descriptor[i] = descriptor[i] / norm;
  }
}

/*************************该函数计算每个特征点的特征描述子*****************************/
/*amplit表示特征点所在层的梯度幅度图像
  orient表示特征点所在层的梯度角度图像
  pt表示特征点的位置
  scale表示特征点所在层的尺度
  main_ori表示特征点的主方向，0-360度
  d表示GLOH角度方向区间个数，默认是8，
  n表示每个网格内角度在0-360度之间等分个数，n默认是8
 */
static void calc_gloh_descriptor(const Mat &amplit, const Mat &orient,
                                 Point2f pt, float scale, float main_ori, int d,
                                 int n, float *ptr_des) {
  Point point(cvRound(pt.x), cvRound(pt.y));

  //特征点旋转方向余弦和正弦
  float cos_t = cosf(-main_ori / 180.f * (float)CV_PI);
  float sin_t = sinf(-main_ori / 180.f * (float)CV_PI);

  int num_rows = amplit.rows;
  int num_cols = amplit.cols;
  int radius = cvRound(SAR_SIFT_RADIUS_DES * scale);
  radius = min(radius, min(num_rows / 2, num_cols / 2)); //特征点邻域半径

  int radius_x_left = point.x - radius;
  int radius_x_right = point.x + radius;
  int radius_y_up = point.y - radius;
  int radius_y_down = point.y + radius;

  //防止越界
  if (radius_x_left < 0)
    radius_x_left = 0;
  if (radius_x_right > num_cols - 1)
    radius_x_right = num_cols - 1;
  if (radius_y_up < 0)
    radius_y_up = 0;
  if (radius_y_down > num_rows - 1)
    radius_y_down = num_rows - 1;

  //此时特征点周围本矩形区域的中心，相对于该矩形
  int center_x = point.x - radius_x_left;
  int center_y = point.y - radius_y_up;

  //特征点周围区域内像素梯度幅度，梯度角度
  Mat sub_amplit = amplit(Range(radius_y_up, radius_y_down + 1),
                          Range(radius_x_left, radius_x_right + 1));
  Mat sub_orient = orient(Range(radius_y_up, radius_y_down + 1),
                          Range(radius_x_left, radius_x_right + 1));

  //以center_x和center_y位中心，对下面矩形区域进行旋转
  Range x_rng(-(point.x - radius_x_left), radius_x_right - point.x);
  Range y_rng(-(point.y - radius_y_up), radius_y_down - point.y);
  Mat X, Y;
  meshgrid(x_rng, y_rng, X, Y);
  Mat c_rot = X * cos_t - Y * sin_t;
  Mat r_rot = X * sin_t + Y * cos_t;
  Mat GLOH_angle, GLOH_amplit;
  phase(c_rot, r_rot, GLOH_angle, true); //角度在0-360度之间
  GLOH_amplit =
      c_rot.mul(c_rot) + r_rot.mul(r_rot); //为了加快速度，没有计算开方

  //三个圆半径平方
  float R1_pow = (float)radius * radius; //外圆半径平方
  float R2_pow = pow(radius * SAR_SIFT_GLOH_RATIO_R1_R2, 2.f); //中间圆半径平方
  float R3_pow = pow(radius * SAR_SIFT_GLOH_RATIO_R1_R3, 2.f); //内圆半径平方

  int sub_rows = sub_amplit.rows;
  int sub_cols = sub_amplit.cols;

  //开始构建描述子,在角度方向对描述子进行插值
  int len = (d * 2 + 1) * n;
  AutoBuffer<float> hist(len);
  for (int i = 0; i < len; ++i) //清零
    hist[i] = 0;

  for (int i = 0; i < sub_rows; ++i) {
    float *ptr_amplit = sub_amplit.ptr<float>(i);
    float *ptr_orient = sub_orient.ptr<float>(i);
    float *ptr_GLOH_amp = GLOH_amplit.ptr<float>(i);
    float *ptr_GLOH_ang = GLOH_angle.ptr<float>(i);
    for (int j = 0; j < sub_cols; ++j) {
      if (((i - center_y) * (i - center_y) + (j - center_x) * (j - center_x)) <
          radius * radius) {
        float pix_amplit = ptr_amplit[j]; //该像素的梯度幅度
        float pix_orient = ptr_orient[j]; //该像素的梯度方向
        float pix_GLOH_amp = ptr_GLOH_amp[j]; //该像素在GLOH网格中的半径位置
        float pix_GLOH_ang = ptr_GLOH_ang[j]; //该像素在GLOH网格中的位置方向

        int rbin, cbin, obin;
        rbin = pix_GLOH_amp < R3_pow
                   ? 0
                   : (pix_GLOH_amp > R2_pow ? 2 : 1); // rbin={0,1,2}
        cbin = cvRound(pix_GLOH_ang * d / 360.f);
        cbin =
            cbin > d ? cbin - d : (cbin <= 0 ? cbin + d : cbin); // cbin=[1,d]

        obin = cvRound(pix_orient * n / 360.f);
        obin =
            obin > n ? obin - n : (obin <= 0 ? obin + n : obin); // obin=[1,n]

        if (rbin == 0) //内圆
          hist[obin - 1] += pix_amplit;
        else {
          int idx = ((rbin - 1) * d + cbin - 1) * n + n + obin - 1;
          hist[idx] += pix_amplit;
        }
      }
    }
  }

  //对描述子进行归一化
  float norm = 0;
  for (int i = 0; i < len; ++i) {
    norm = norm + hist[i] * hist[i];
  }
  norm = sqrt(norm);
  for (int i = 0; i < len; ++i) {
    hist[i] = hist[i] / norm;
  }

  //阈值截断
  for (int i = 0; i < len; ++i) {
    hist[i] = min(hist[i], DESCR_MAG_THR);
  }

  //再次归一化
  norm = 0;
  for (int i = 0; i < len; ++i) {
    norm = norm + hist[i] * hist[i];
  }
  norm = sqrt(norm);
  for (int i = 0; i < len; ++i) {
    ptr_des[i] = hist[i] / norm;
  }
}

/******************************计算一个特征点的描述子—改进版***********************************/
static void improve_calc_sift_descriptor(const Mat &gauss_image,
                                         float main_angle, Point2f pt,
                                         float scale, int d, int n,
                                         float *descriptor) {
  int n1 = 16, n2 = 6, n3 = 4;

  Point ptxy(cvRound(pt.x), cvRound(pt.y)); //坐标取整

  float cos_t = cosf(-main_angle * (float)(CV_PI / 180)); //计算主方向的余弦
  float sin_t = sinf(-main_angle * (float)(CV_PI / 180)); //计算主方向的正弦

  float bin_per_rad_1 = n1 / 360.f; // n=8
  float bin_per_rad_2 = n2 / 360.f; //原理特征点部分阈值
  float bin_per_rad_3 = n3 / 360.f; //原理特征点部分阈值

  float exp_scale = -1.f / (d * d * 0.5f); //权重指数部分
  float hist_width =
      DESCR_SCL_FCTR * scale; //子区域边长，子区域的面积也即采样像素点个数

  int radius = cvRound(hist_width * (d + 1) * sqrt(2) *
                       0.5f); //特征点邻域半径(d+1)*(d+1)

  int rows = gauss_image.rows, cols = gauss_image.cols;

  //特征点邻域半径
  radius = min(radius, (int)sqrt((double)rows * rows + cols * cols));

  cos_t = cos_t / hist_width;
  sin_t = sin_t / hist_width;

  int len = (2 * radius + 1) * (2 * radius + 1); //邻域内总像素数
  int histlen = (d + 2) * (d + 2) * (n1 + 2);

  AutoBuffer<float> buf(6 * len + histlen);

  // X保存水平差分，Y保存竖直差分，Mag保存梯度幅度，Angle保存特征点方向,
  // W保存高斯权重
  float *X = buf, *Y = buf + len, *Mag = Y, *Angle = Y + len, *W = Angle + len;
  float *X2 = buf, *Y2 = buf + len, *Mag2 = Y, *Angle2 = Y + len,
        *W2 = Angle + len;
  float *X3 = buf, *Y3 = buf + len, *Mag3 = Y, *Angle3 = Y + len,
        *W3 = Angle + len;

  float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;
  float *RBin2 = W + len, *CBin2 = RBin + len;
  float *RBin3 = W + len, *CBin3 = RBin + len;

  //首先清空直方图数据
  for (int i = 0; i < d + 2; ++i) {
    for (int j = 0; j < d + 2; ++j) {
      for (int k = 0; k < n + 2; ++k)
        hist[(i * (d + 2) + j) * (n + 2) + k] = 0.f;
    }
  }

  //把邻域内的像素分配到相应子区域内，计算子区域内每个像素点的权重(子区域即 d*d
  //中每一个小方格)

  int k1 = 0, k2 = 0, k3 = 0;

  vector<int> v; //存放外邻域像素点对应的序号

  for (int i = -radius; i < radius; ++i) {
    for (int j = -radius; j < radius; ++j) {
      float c_rot = j * cos_t - i * sin_t; //旋转后邻域内采样点的 x 坐标
      float r_rot = j * sin_t + i * cos_t; //旋转后邻域内采样点的 y 坐标

      float rbin = r_rot + d / 2 - 0.5f; //旋转后的采样点落在子区域的 y 坐标
      float cbin = c_rot + d / 2 - 0.5f; //旋转后的采样点落在子区域的 x 坐标

      int r = ptxy.y + i, c = ptxy.x + j; // ptxy是高斯金字塔中的坐标

      //对离中心点近的部分进行操作
      if (abs(i) < (radius / 3) && abs(j) < (radius / 3)) {
        //这里rbin,cbin范围是(-1,d)
        if (rbin > -1 && rbin < d && cbin > -1 && cbin < d && r > 0 &&
            r < rows - 1 && c > 0 && c < cols - 1) {
          float dx =
              gauss_image.at<float>(r, c + 1) - gauss_image.at<float>(r, c - 1);
          float dy =
              gauss_image.at<float>(r + 1, c) - gauss_image.at<float>(r - 1, c);

          X[k1] = dx; //邻域内所有像素点的水平差分
          Y[k1] = dy; //邻域内所有像素点的竖直差分

          RBin[k1] = rbin; //邻域内所有采样点落在子区域的 y 坐标
          CBin[k1] = cbin; //邻域内所有采样点落在子区域的 x 坐标

          //高斯权值的指数部分
          W[k1] = (c_rot * c_rot + r_rot * r_rot) * exp_scale;

          ++k1;
        }
      }
      //对离中心点远的部分进行操作
      else if (abs(i) < (2 * radius / 3) && abs(i) > (radius / 3) &&
               abs(j) < (2 * radius / 3) && abs(j) > (radius / 3)) {
        //这里rbin,cbin范围是(-1,d)
        if (rbin > -1 && rbin < d && cbin > -1 && cbin < d && r > 0 &&
            r < rows - 1 && c > 0 && c < cols - 1) {
          float dx =
              gauss_image.at<float>(r, c + 1) - gauss_image.at<float>(r, c - 1);
          float dy =
              gauss_image.at<float>(r + 1, c) - gauss_image.at<float>(r - 1, c);

          X2[k2] = dx; //邻域内所有像素点的水平差分
          Y2[k2] = dy; //邻域内所有像素点的竖直差分

          RBin2[k2] = rbin; //邻域内所有采样点落在子区域的 y 坐标
          CBin2[k2] = cbin; //邻域内所有采样点落在子区域的 x 坐标

          //高斯权值的指数部分
          W2[k2] = (c_rot * c_rot + r_rot * r_rot) * exp_scale;

          ++k2;
        }
      } else {
        //这里rbin,cbin范围是(-1,d)
        if (rbin > -1 && rbin < d && cbin > -1 && cbin < d && r > 0 &&
            r < rows - 1 && c > 0 && c < cols - 1) {
          float dx =
              gauss_image.at<float>(r, c + 1) - gauss_image.at<float>(r, c - 1);
          float dy =
              gauss_image.at<float>(r + 1, c) - gauss_image.at<float>(r - 1, c);

          X3[k3] = dx; //邻域内所有像素点的水平差分
          Y3[k3] = dy; //邻域内所有像素点的竖直差分

          RBin3[k3] = rbin; //邻域内所有采样点落在子区域的 y 坐标
          CBin3[k3] = cbin; //邻域内所有采样点落在子区域的 x 坐标

          //高斯权值的指数部分
          W3[k3] = (c_rot * c_rot + r_rot * r_rot) * exp_scale;

          ++k3;
        }
      }
    }
  }

  //两个区域内数组的合并拼接
  for (int k = 0; k < k2; k++) {
    X[k1 + k] = X2[k];
    Y[k1 + k] = Y2[k];

    RBin[k1 + k] = RBin2[k];
    CBin[k1 + k] = CBin2[k];

    W[k1 + k] = W2[k];
  }

  for (int k = 0; k < k3; k++) {
    X[k1 + k2 + k] = X3[k];
    Y[k1 + k2 + k] = Y3[k];

    RBin[k1 + k2 + k] = RBin3[k];
    CBin[k1 + k2 + k] = CBin3[k];

    W[k1 + k2 + k] = W3[k];
  }

  //计算采样点落在子区域的像素梯度幅度，梯度角度，和高斯权值
  len = k1 + k2 + k3;

  cv::hal::exp(W, W, len); //邻域内所有采样点落在子区域的像素的高斯权值
  cv::hal::fastAtan2(
      Y, X, Angle, len,
      true); //邻域内所有采样点落在子区域的像素的梯度方向，角度范围是0-360度
  cv::hal::magnitude(X, Y, Mag,
                     len); //邻域内所有采样点落在子区域的像素的梯度幅度

  //计算每个特征点的特征描述子
  for (int k = 0; k < len; ++k) {
    float rbin = RBin[k],
          cbin = CBin[k]; //子区域内像素点坐标，rbin,cbin范围是(-1,d)

    //离特征点进的邻域
    if (k < k1) {
      //子区域内像素点处理后的方向
      float obin = (Angle[k] - main_angle) * bin_per_rad_1;

      float mag = Mag[k] * W[k]; //子区域内像素点乘以权值后的梯度幅值

      int r0 = cvFloor(rbin); // ro取值集合是{-1,0,1,2，3}，向下取整
      int c0 = cvFloor(cbin); // c0取值集合是{-1，0，1，2，3}
      int o0 = cvFloor(obin);

      rbin = rbin - r0; //子区域内像素点坐标的小数部分，用于线性插值
      cbin = cbin - c0;
      obin = obin - o0;

      //限制范围为梯度直方图横坐标[0,n)
      if (o0 < 0)
        o0 = o0 + n1;
      if (o0 >= n1)
        o0 = o0 - n1;

      //三线性插值用于计算落在两个子区域之间的像素对两个子区域的作用，并把其分配到对应子区域的8个方向上
      //使用三线性插值(即三维)方法，计算直方图
      float v_r1 = mag * rbin; //第二行分配的值
      float v_r0 = mag - v_r1; //第一行分配的值

      float v_rc11 = v_r1 * cbin;   //第二行第二列分配的值
      float v_rc10 = v_r1 - v_rc11; //第二行第一列分配的值

      float v_rc01 = v_r0 * cbin; //第一行第二列分配的值
      float v_rc00 = v_r0 - v_rc01;

      float v_rco111 =
          v_rc11 *
          obin; //第二行第二列第二个方向上分配的值，每个采样点去邻近两个方向
      float v_rco110 = v_rc11 - v_rco111; //第二行第二列第一个方向上分配的值

      float v_rco101 = v_rc10 * obin;
      float v_rco100 = v_rc10 - v_rco101;

      float v_rco011 = v_rc01 * obin;
      float v_rco010 = v_rc01 - v_rco011;

      float v_rco001 = v_rc00 * obin;
      float v_rco000 = v_rc00 - v_rco001;

      //该像素所在网格的索引
      int idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n1 + 2) + o0;

      hist[idx] += v_rco000;
      hist[idx + 1] += v_rco001;
      hist[idx + n1 + 2] += v_rco010;
      hist[idx + n1 + 3] += v_rco011;
      hist[idx + (d + 2) * (n1 + 2)] += v_rco100;
      hist[idx + (d + 2) * (n1 + 2) + 1] += v_rco101;
      hist[idx + (d + 3) * (n1 + 2)] += v_rco110;
      hist[idx + (d + 3) * (n1 + 2) + 1] += v_rco111;
    }

    //离特征点远的邻域
    else if (k >= k1 && k < k2) {
      //子区域内像素点处理后的方向
      float obin = (Angle[k] - main_angle) * bin_per_rad_2;

      float mag = Mag[k] * W[k]; //子区域内像素点乘以权值后的梯度幅值

      int r0 = cvFloor(rbin); // ro取值集合是{-1,0,1,2，3}，向下取整
      int c0 = cvFloor(cbin); // c0取值集合是{-1，0，1，2，3}
      int o0 = cvFloor(obin);

      rbin = rbin - r0; //子区域内像素点坐标的小数部分，用于线性插值
      cbin = cbin - c0;
      obin = obin - o0;

      //限制范围为梯度直方图横坐标[0,n)
      if (o0 < 0)
        o0 = o0 + n2;
      if (o0 >= n1)
        o0 = o0 - n2;

      //三线性插值用于计算落在两个子区域之间的像素对两个子区域的作用，并把其分配到对应子区域的8个方向上
      //使用三线性插值(即三维)方法，计算直方图
      float v_r1 = mag * rbin; //第二行分配的值
      float v_r0 = mag - v_r1; //第一行分配的值

      float v_rc11 = v_r1 * cbin;   //第二行第二列分配的值
      float v_rc10 = v_r1 - v_rc11; //第二行第一列分配的值

      float v_rc01 = v_r0 * cbin; //第一行第二列分配的值
      float v_rc00 = v_r0 - v_rc01;

      float v_rco111 =
          v_rc11 *
          obin; //第二行第二列第二个方向上分配的值，每个采样点去邻近两个方向
      float v_rco110 = v_rc11 - v_rco111; //第二行第二列第一个方向上分配的值

      float v_rco101 = v_rc10 * obin;
      float v_rco100 = v_rc10 - v_rco101;

      float v_rco011 = v_rc01 * obin;
      float v_rco010 = v_rc01 - v_rco011;

      float v_rco001 = v_rc00 * obin;
      float v_rco000 = v_rc00 - v_rco001;

      //该像素所在网格的索引
      int idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n2 + 2) + o0;

      hist[idx] += v_rco000;
      hist[idx + 1] += v_rco001;
      hist[idx + n2 + 2] += v_rco010;
      hist[idx + n2 + 3] += v_rco011;
      hist[idx + (d + 2) * (n2 + 2)] += v_rco100;
      hist[idx + (d + 2) * (n2 + 2) + 1] += v_rco101;
      hist[idx + (d + 3) * (n2 + 2)] += v_rco110;
      hist[idx + (d + 3) * (n2 + 2) + 1] += v_rco111;
    } else {
      //子区域内像素点处理后的方向
      float obin = (Angle[k] - main_angle) * bin_per_rad_3;

      float mag = Mag[k] * W[k]; //子区域内像素点乘以权值后的梯度幅值

      int r0 = cvFloor(rbin); // ro取值集合是{-1,0,1,2，3}，向下取整
      int c0 = cvFloor(cbin); // c0取值集合是{-1，0，1，2，3}
      int o0 = cvFloor(obin);

      rbin = rbin - r0; //子区域内像素点坐标的小数部分，用于线性插值
      cbin = cbin - c0;
      obin = obin - o0;

      //限制范围为梯度直方图横坐标[0,n)
      if (o0 < 0)
        o0 = o0 + n3;
      if (o0 >= n1)
        o0 = o0 - n3;

      //三线性插值用于计算落在两个子区域之间的像素对两个子区域的作用，并把其分配到对应子区域的8个方向上
      //使用三线性插值(即三维)方法，计算直方图
      float v_r1 = mag * rbin; //第二行分配的值
      float v_r0 = mag - v_r1; //第一行分配的值

      float v_rc11 = v_r1 * cbin;   //第二行第二列分配的值
      float v_rc10 = v_r1 - v_rc11; //第二行第一列分配的值

      float v_rc01 = v_r0 * cbin; //第一行第二列分配的值
      float v_rc00 = v_r0 - v_rc01;

      float v_rco111 =
          v_rc11 *
          obin; //第二行第二列第二个方向上分配的值，每个采样点去邻近两个方向
      float v_rco110 = v_rc11 - v_rco111; //第二行第二列第一个方向上分配的值

      float v_rco101 = v_rc10 * obin;
      float v_rco100 = v_rc10 - v_rco101;

      float v_rco011 = v_rc01 * obin;
      float v_rco010 = v_rc01 - v_rco011;

      float v_rco001 = v_rc00 * obin;
      float v_rco000 = v_rc00 - v_rco001;

      //该像素所在网格的索引
      int idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n3 + 2) + o0;

      hist[idx] += v_rco000;
      hist[idx + 1] += v_rco001;
      hist[idx + n3 + 2] += v_rco010;
      hist[idx + n3 + 3] += v_rco011;
      hist[idx + (d + 2) * (n3 + 2)] += v_rco100;
      hist[idx + (d + 2) * (n3 + 2) + 1] += v_rco101;
      hist[idx + (d + 3) * (n3 + 2)] += v_rco110;
      hist[idx + (d + 3) * (n3 + 2) + 1] += v_rco111;
    }
  }

  //由于圆周循环的特性，对计算以后幅角小于 0 度或大于 360 度的值重新进行调整，使
  //其在 0～360 度之间
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < d; ++j) {
      int idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);
      hist[idx] += hist[idx + n];
      // hist[idx + 1] += hist[idx + n +
      // 1];//opencv源码中这句话是多余的,hist[idx + n + 1]永远是0.0
      for (int k = 0; k < n; ++k)
        descriptor[(i * d + j) * n + k] = hist[idx + k];
    }
  }

  //对描述子进行归一化
  int lenght = d * d * n;
  float norm = 0;

  //计算特征描述向量的模值的平方
  for (int i = 0; i < lenght; ++i) {
    norm = norm + descriptor[i] * descriptor[i];
  }
  norm = sqrt(norm); //特征描述向量的模值

  //此次归一化能去除光照的影响
  for (int i = 0; i < lenght; ++i) {
    descriptor[i] = descriptor[i] / norm;
  }

  //阈值截断,去除特征描述向量中大于 0.2
  //的值，能消除非线性光照的影响(相机饱和度对某些放的梯度影响较大，对方向的影响较小)
  for (int i = 0; i < lenght; ++i) {
    descriptor[i] = min(descriptor[i], DESCR_MAG_THR);
  }

  //再次归一化，能够提高特征的独特性
  norm = 0;

  for (int i = 0; i < lenght; ++i) {
    norm = norm + descriptor[i] * descriptor[i];
  }
  norm = sqrt(norm);
  for (int i = 0; i < lenght; ++i) {
    descriptor[i] = descriptor[i] / norm;
  }
}

/********************************该函数计算所有特征点特征描述子***************************/
/*gauss_pyr表示高斯金字塔
  keypoints表示特征点、
  descriptors表示生成的特征点的描述子*/
void mySift::calc_sift_descriptors(const vector<vector<Mat>> &gauss_pyr,
                                   vector<KeyPoint> &keypoints,
                                   Mat &descriptors, const vector<Mat> &amplit,
                                   const vector<Mat> &orient) {
  int d = DESCR_WIDTH; // d=4,特征点邻域网格个数是d x d
  int n = DESCR_HIST_BINS; // n=8,每个网格特征点梯度角度等分为8个方向

  descriptors.create(keypoints.size(), d * d * n, CV_32FC1); //分配空间

  for (size_t i = 0; i < keypoints.size(); ++i) //对于每一个特征点
  {
    int octaves, layer;

    //得到特征点所在的组号，层号
    octaves = keypoints[i].octave & 255;
    layer = (keypoints[i].octave >> 8) & 255;

    //得到特征点相对于本组的坐标，不是最底层
    Point2f pt(keypoints[i].pt.x / (1 << octaves),
               keypoints[i].pt.y / (1 << octaves));

    float scale =
        keypoints[i].size / (1 << octaves); //得到特征点相对于本组的尺度
    float main_angle = keypoints[i].angle; //特征点主方向

    //计算该点的描述子
    calc_sift_descriptor(gauss_pyr[octaves][layer], main_angle, pt, scale, d, n,
                         descriptors.ptr<float>((int)i));
    // improve_calc_sift_descriptor(gauss_pyr[octaves][layer], main_angle, pt,
    // scale, d, n, descriptors.ptr<float>((int)i));
    // calc_gloh_descriptor(amplit[octaves], orient[octaves], pt, scale,
    // main_angle, d, n, descriptors.ptr<float>((int)i));

    if (double_size) //如果图像尺寸扩大一倍
    {
      keypoints[i].pt.x = keypoints[i].pt.x / 2.f;
      keypoints[i].pt.y = keypoints[i].pt.y / 2.f;
    }
  }
}

/*************该函数构建SAR_SIFT尺度空间*****************/
/*image表示输入的原始图像
 sar_harris_fun表示尺度空间的Sar_harris函数
 amplit表示尺度空间像素的梯度幅度
 orient表示尺度空间像素的梯度方向
 */
void mySift::build_sar_sift_space(const Mat &image, vector<Mat> &sar_harris_fun,
                                  vector<Mat> &amplit, vector<Mat> &orient) {
  //转换输入图像格式
  Mat gray_image;
  if (image.channels() != 1)
    cvtColor(image, gray_image, CV_RGB2GRAY);
  else
    gray_image = image;

  //把图像转换为0-1之间的浮点数据
  Mat float_image;

  double ratio = pow(2, 1.0 / 3.0); //相邻两层的尺度比,默认是2^(1/3)

  //在这里转换为0-1之间的浮点数据和转换为0-255之间的浮点数据，效果是一样的
  // gray_image.convertTo(float_image, CV_32FC1, 1.f / 255.f,
  // 0.f);//转换为0-1之间

  gray_image.convertTo(float_image, CV_32FC1, 1, 0.f); //转换为0-255之间的浮点数

  //分配内存
  sar_harris_fun.resize(Mmax);
  amplit.resize(Mmax);
  orient.resize(Mmax);

  for (int i = 0; i < Mmax; ++i) {
    float scale = (float)sigma * (float)pow(ratio, i); //获得当前层的尺度
    int radius = cvRound(2 * scale);
    Mat kernel;

    roewa_kernel(radius, scale, kernel);

    //四个滤波模板生成
    Mat W34 = Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_32FC1);
    Mat W12 = Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_32FC1);
    Mat W14 = Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_32FC1);
    Mat W23 = Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_32FC1);

    kernel(Range(radius + 1, 2 * radius + 1), Range::all())
        .copyTo(W34(Range(radius + 1, 2 * radius + 1), Range::all()));
    kernel(Range(0, radius), Range::all())
        .copyTo(W12(Range(0, radius), Range::all()));
    kernel(Range::all(), Range(radius + 1, 2 * radius + 1))
        .copyTo(W14(Range::all(), Range(radius + 1, 2 * radius + 1)));
    kernel(Range::all(), Range(0, radius))
        .copyTo(W23(Range::all(), Range(0, radius)));

    //滤波
    Mat M34, M12, M14, M23;
    double eps = 0.00001;
    filter2D(float_image, M34, CV_32FC1, W34, Point(-1, -1), eps);
    filter2D(float_image, M12, CV_32FC1, W12, Point(-1, -1), eps);
    filter2D(float_image, M14, CV_32FC1, W14, Point(-1, -1), eps);
    filter2D(float_image, M23, CV_32FC1, W23, Point(-1, -1), eps);

    //计算水平梯度和竖直梯度
    Mat Gx, Gy;
    log((M14) / (M23), Gx);
    log((M34) / (M12), Gy);

    //计算梯度幅度和梯度方向
    magnitude(Gx, Gy, amplit[i]);
    phase(Gx, Gy, orient[i], true);

    //构建sar-Harris矩阵
    // Mat Csh_11 = log(scale)*log(scale)*Gx.mul(Gx);
    // Mat Csh_12 = log(scale)*log(scale)*Gx.mul(Gy);
    // Mat Csh_22 = log(scale)*log(scale)*Gy.mul(Gy);

    Mat Csh_11 = scale * scale * Gx.mul(Gx);
    Mat Csh_12 = scale * scale * Gx.mul(Gy);
    Mat Csh_22 = scale * scale * Gy.mul(Gy); //此时阈值为0.8

    // Mat Csh_11 = Gx.mul(Gx);
    // Mat Csh_12 = Gx.mul(Gy);
    // Mat Csh_22 = Gy.mul(Gy);//此时阈值为0.8/100

    //高斯权重
    float gauss_sigma = sqrt(2.f) * scale;
    int size = cvRound(3 * gauss_sigma);

    Size kern_size(2 * size + 1, 2 * size + 1);
    GaussianBlur(Csh_11, Csh_11, kern_size, gauss_sigma, gauss_sigma);
    GaussianBlur(Csh_12, Csh_12, kern_size, gauss_sigma, gauss_sigma);
    GaussianBlur(Csh_22, Csh_22, kern_size, gauss_sigma, gauss_sigma);

    /*Mat gauss_kernel;//自定义圆形高斯核
    gauss_circle(size, gauss_sigma, gauss_kernel);
    filter2D(Csh_11, Csh_11, CV_32FC1, gauss_kernel);
    filter2D(Csh_12, Csh_12, CV_32FC1, gauss_kernel);
    filter2D(Csh_22, Csh_22, CV_32FC1, gauss_kernel);*/

    Mat Csh_21 = Csh_12;

    //构建sar_harris函数
    Mat temp_add = Csh_11 + Csh_22;

    double d = 0.04; // sar_haiirs函数表达式中的任意参数，默认是0.04

    sar_harris_fun[i] = Csh_11.mul(Csh_22) - Csh_21.mul(Csh_12) -
                        (float)d * temp_add.mul(temp_add);
  }
}

/***************该函数计算所有特征点的特征向量*************/
/*amplit表示尺度空间像素幅度
 orient表示尺度空间像素梯度角度
 keys表示检测到的特征点
 descriptors表示特征点描述子向量，【M x N】,M表示描述子个数，N表示描述子维度
 */
void mySift::calc_gloh_descriptors(const vector<Mat> &amplit,
                                   const vector<Mat> &orient,
                                   const vector<KeyPoint> &keys,
                                   Mat &descriptors) {
  int d = SAR_SIFT_GLOH_ANG_GRID; // d=4或者d=8
  int n = SAR_SIFT_DES_ANG_BINS;  // n=8默认

  int num_keys = (int)keys.size();
  int grids = 2 * d + 1;

  // descriptors.create(num_keys, grids * n, CV_32FC1);
  descriptors.create(num_keys, grids * n, CV_32FC1);

  for (int i = 0; i < num_keys; ++i) {
    int octaves = keys[i].octave & 255; //特征点所在层

    float *ptr_des = descriptors.ptr<float>(i);
    float scale = keys[i].size / (1 << octaves); //得到特征点相对于本组的尺度;
                                                 ////特征点所在层的尺度
    float main_ori = keys[i].angle; //特征点主方向

    //得到特征点相对于本组的坐标，不是最底层
    Point2f point(keys[i].pt.x / (1 << octaves), keys[i].pt.y / (1 << octaves));

    cout << "layer=" << octaves << endl;
    cout << "scale=" << scale << endl;

    //计算该特征点的特征描述子
    calc_gloh_descriptor(amplit[octaves], orient[octaves], point, scale,
                         main_ori, d, n, ptr_des);
  }
}

//特征点检测和特征点描述把整个 SIFT 算子都涵盖在内了//

/******************************特征点检测*********************************/
/*image表示输入的图像
  gauss_pyr表示生成的高斯金字塔
  dog_pyr表示生成的高斯差分DOG金字塔
  keypoints表示检测到的特征点
  vector<float>& cell_contrast 用于存放一个单元格中所有特征点的对比度
  vector<float>& cell_contrasts用于存放一个尺度层中所有单元格中特征点的对比度
  vector<vector<vector<float>>>&
  all_cell_contrasts用于存放所有尺度层中所有单元格的对比度
  vector<vector<float>>&
  average_contrast用于存放所有尺度层中多有单元格的平均对比度*/
void mySift::detect(const Mat &image, vector<vector<Mat>> &gauss_pyr,
                    vector<vector<Mat>> &dog_pyr, vector<KeyPoint> &keypoints,
                    vector<vector<vector<float>>> &all_cell_contrasts,
                    vector<vector<float>> &average_contrast,
                    vector<vector<int>> &n_cells, vector<int> &num_cell,
                    vector<vector<int>> &available_n,
                    vector<int> &available_num,
                    vector<KeyPoint> &final_keypoints,
                    vector<KeyPoint> &Final_keypoints,
                    vector<KeyPoint> &Final_Keypoints) {
  if (image.empty() || image.depth() != CV_8U)
    CV_Error(CV_StsBadArg, "输入图像为空，或者图像深度不是CV_8U");

  //计算高斯金字塔组数
  int nOctaves;
  nOctaves = num_octaves(image);

  //生成高斯金字塔第一层图像
  Mat init_gauss;
  create_initial_image(image, init_gauss);
  // ---------------------------------------------------
  int _o = 0;
  // ---------------------------------------------------
  //生成高斯尺度空间图像
  build_gaussian_pyramid(init_gauss, gauss_pyr, nOctaves);
  // Mat tmp = Mat(gauss_pyr[0][0].size(), CV_8U);
  // cout << "gauss_pyr[0][0].size()=" << gauss_pyr[0][0].size() << endl;
  // cout << gauss_pyr[0][0].channels() << endl;
  // cout << gauss_pyr[0][0].depth() << endl;
  // cout << gauss_pyr[0][0].type() << endl;

  // for (int s = 0; s < gauss_pyr[_o].size(); s++) {
  //   MY_IMG::ConvertTo<float, uint8_t>(
  //       gauss_pyr[_o][s], tmp,
  //       [](const float &x) -> uint8_t { return x * 255; });
  //   imshow("gauss_pyr-" + to_string(s), tmp);
  // }
  //生成高斯差分金字塔(DOG金字塔，or LOG金字塔)
  build_dog_pyramid(dog_pyr, gauss_pyr);
  // for (int s = 0; s < dog_pyr[_o].size(); s++) {
  //   MY_IMG::ConvertTo<float, uint8_t>(
  //       dog_pyr[_o][s], tmp, [](const float &x) -> uint8_t { return x * 255;
  //       });
  //   imshow("dog_pyr-" + to_string(s), tmp);
  // }
  // waitKey(0);
  //在DOG金字塔上检测特征点
  find_scale_space_extrema1(dog_pyr, gauss_pyr, keypoints);
  //原始 SIFT 算子
  // UR_SIFT,仅仅是检测特征点，并未对不理想的点进行筛选
  /*UR_SIFT_find_scale_space_extrema(dog_pyr, gauss_pyr, keypoints,
  all_cell_contrasts, average_contrast, n_cells, num_cell, available_n,
  available_num, final_keypoints, Final_keypoints, Final_Keypoints, N);*/

  //如果指定了特征点个数,并且设定的设置小于默认检测的特征点个数
  if (nfeatures != 0 && nfeatures < (int)keypoints.size()) {
    //特征点响应值(即对比度，对比度越小越不稳定)从大到小排序
    sort(keypoints.begin(), keypoints.end(),
         [](const KeyPoint &a, const KeyPoint &b) {
           return a.response > b.response;
         });

    //删除点多余的特征点
    keypoints.erase(keypoints.begin() + nfeatures, keypoints.end());
  }
}

/**********************特征点描述*******************/
/*gauss_pyr表示高斯金字塔
 keypoints表示特征点集合
 descriptors表示特征点的描述子*/
void mySift::comput_des(const vector<vector<Mat>> &gauss_pyr,
                        vector<KeyPoint> &final_keypoints,
                        const vector<Mat> &amplit, const vector<Mat> &orient,
                        Mat &descriptors) {
  calc_sift_descriptors(gauss_pyr, final_keypoints, descriptors, amplit,
                        orient);
}
