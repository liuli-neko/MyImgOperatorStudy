#include "unit.h"
#include <ctime>
#include <iostream>

using namespace MY_IMG;

int main(int argc, char **argv) {
  std::string file_name = "./test_img.png";
  std::string output_name = "./test_img_out.png";
  if (argc >= 2) {
    file_name = std::string(argv[1]);
    if (argc >= 3) {
      output_name = std::string(argv[2]);
    }
  }
  cv::Mat img = cv::imread(file_name, cv::IMREAD_COLOR);
  // get dft
  cv::Mat dft_img;
  cv::Mat gray_img;
  Rbg2Gray(img, gray_img);
  cv::imshow("gray_img", gray_img);
  time_t t1 = clock();
  DFT(gray_img, dft_img);
  time_t t2 = clock();
  LOG("DFT time: %f", static_cast<double>(t2 - t1) / 1000.0);
  // show
  cv::imshow("dft_img", ConvertComplexMat2doubleMat(dft_img));

  // 创建巴特沃斯滤波器
  auto butter_filter =
      ButterworthFilter({gray_img.rows / 2, gray_img.cols / 2});
  cv::Mat butter_img_low(dft_img.size(), dft_img.type()),
      butter_img_high(dft_img.size(), dft_img.type()),
      butter_img_band(dft_img.size(), dft_img.type());
  // 进行滤波
  for (int i = 0; i < gray_img.rows; i++) {
    for (int j = 0; j < gray_img.cols; j++) {
      butter_img_low.at<std::complex<double>>(i, j) =
          dft_img.at<std::complex<double>>(i, j) *
          butter_filter.LowPassFilter(i, j);
    }
  }
  butter_filter.D0 = 80;
  butter_filter.n = 2;
  for (int i = 0; i < gray_img.rows; i++) {
    for (int j = 0; j < gray_img.cols; j++) {
      butter_img_high.at<std::complex<double>>(i, j) =
          dft_img.at<std::complex<double>>(i, j) *
          butter_filter.HighPassFilter(i, j);
    }
  }
  for (int i = 0; i < gray_img.rows; i++) {
    for (int j = 0; j < gray_img.cols; j++) {
      butter_img_band.at<std::complex<double>>(i, j) =
          dft_img.at<std::complex<double>>(i, j) *
          butter_filter.BandPassFilter(i, j);
    }
  }

  // 创建高斯滤波器
  auto gauss_filter =
      GaussianFilter({gray_img.rows / 2, gray_img.cols / 2}, 10);
  cv::Mat gauss_img_low(dft_img.size(), dft_img.type()),
      gauss_img_high(dft_img.size(), dft_img.type());
  // 进行滤波
  for (int i = 0; i < gray_img.rows; i++) {
    for (int j = 0; j < gray_img.cols; j++) {
      gauss_img_low.at<std::complex<double>>(i, j) =
          dft_img.at<std::complex<double>>(i, j) *
          gauss_filter.LowPassFilter(i, j);
    }
  }
  gauss_filter.D0 = 60;
  for (int i = 0; i < gray_img.rows; i++) {
    for (int j = 0; j < gray_img.cols; j++) {
      gauss_img_high.at<std::complex<double>>(i, j) =
          dft_img.at<std::complex<double>>(i, j) *
          gauss_filter.HighPassFilter(i, j);
    }
  }

  // get idft
  cv::Mat idft_img;
  t1 = clock();
  IDFT(dft_img, idft_img);
  t2 = clock();
  LOG("IDFT time: %f", static_cast<double>(t2 - t1) / 1000.0);
  // show
  cv::imshow("idft_img", idft_img);

  // get butterworth low pass
  cv::Mat butter_low_img;
  t1 = clock();
  IDFT(butter_img_low, butter_low_img);
  t2 = clock();
  LOG("Butterworth low pass time: %f", static_cast<double>(t2 - t1) / 1000.0);
  // show
  cv::imshow("butter_low_img", butter_low_img);

  // get butterworth high pass
  cv::Mat butter_high_img;
  t1 = clock();
  IDFT(butter_img_high, butter_high_img);
  t2 = clock();
  LOG("Butterworth high pass time: %f", static_cast<double>(t2 - t1) / 1000.0);
  // show
  cv::imshow("butter_high_img", butter_high_img);

  // get butterworth band pass
  cv::Mat butter_band_img;
  t1 = clock();
  IDFT(butter_img_band, butter_band_img);
  t2 = clock();
  LOG("Butterworth band pass time: %f", static_cast<double>(t2 - t1) / 1000.0);
  // show
  cv::imshow("butter_band_img", butter_band_img);

  // get gauss low pass
  cv::Mat gauss_low_img;
  t1 = clock();
  IDFT(gauss_img_low, gauss_low_img);
  t2 = clock();
  LOG("Gauss low pass time: %f", static_cast<double>(t2 - t1) / 1000.0);
  // show
  cv::imshow("gauss_low_img", gauss_low_img);

  // get gauss high pass
  cv::Mat gauss_high_img;
  t1 = clock();
  IDFT(gauss_img_high, gauss_high_img);
  t2 = clock();
  LOG("Gauss high pass time: %f", static_cast<double>(t2 - t1) / 1000.0);
  // show
  cv::imshow("gauss_high_img", gauss_high_img);

  // 拆分通道
  cv::Mat img_one[3];
  cv::split(img, img_one);
  gauss_filter.D0 = 10;
  // 将三个通道单独使用高通滤波器进行滤波
  cv::Mat img_one_dft[3];
  cv::Mat img_one_dft_low[3];
  cv::Mat img_one_low[3];
  for (int channe = 0; channe < 3; channe++) {
    // 单独对每个通道进行傅里叶变换
    DFT(img_one[channe], img_one_dft[channe]);
    img_one_dft_low[channe] =
        cv::Mat(img_one_dft[channe].size(), img_one_dft[channe].type());
    for (int i = 0; i < img_one[channe].rows; i++) {
      for (int j = 0; j < img_one[channe].cols; j++) {
        img_one_dft_low[channe].at<std::complex<double>>(i, j) =
            img_one_dft[channe].at<std::complex<double>>(i, j) *
            gauss_filter.LowPassFilter(i, j);
      }
    }
    // 将三个通道的高通滤波结果进行IDFT
    IDFT(img_one_dft_low[channe], img_one_low[channe]);
  }
  // 将三个通道的高通滤波结果进行合并
  cv::Mat img_one_low_merge;
  cv::merge(img_one_low, 3, img_one_low_merge);
  cv::imshow("img_one_low_merge", img_one_low_merge);

  cv::waitKey(0);
  return 0;
}