#include "image_operator.h"
#include "unit.h"
#include <iostream>
#include <time.h>

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

  cv::Mat gray_img;
  Rgb2Gray(img, gray_img);
  cv::Mat dft_img_abs, idft_img;
  // cv::imshow("gray_img", gray_img);
  /*
  time_t t1,t2;
  cv::Mat fft_img;


  LOG(INFO,"size : [%d %d]", gray_img.cols, gray_img.rows);
  t1 = clock();
  FFT2D(gray_img, fft_img);
  t2 = clock();
  LOG(INFO,"size : [%d %d]", fft_img.cols, fft_img.rows);
  LOG(INFO,"FFT time: %f", static_cast<double>(t2 - t1) / CLOCKS_PER_SEC);
  // show
  dft_img_abs = ConvertComplexMat2doubleMat(fft_img);
  // std::cout << dft_img_abs << std::endl;
  cv::imshow("dft_img",ConvertSingleChannelMat2Uint8Mat<double>(dft_img_abs));

  // 创建巴特沃斯滤波器
  auto butter_filter =
      ButterworthFilter({fft_img.rows / 2, fft_img.cols / 2});
  cv::Mat butter_img_low(fft_img.size(), fft_img.type());
  // 进行滤波
  for (int i = 0; i < fft_img.rows; i++) {
    for (int j = 0; j < fft_img.cols; j++) {
      butter_img_low.at<std::complex<double>>(i, j) =
          fft_img.at<std::complex<double>>(i, j) *
          butter_filter.LowPassFilter(i, j);
    }
  }

  // get butterworth low pass
  cv::Mat butter_low_img;
  t1 = clock();
  IFFT2D(butter_img_low, butter_low_img);
  t2 = clock();
  LOG(INFO,"size: %d, %d", butter_low_img.rows, butter_low_img.cols);
  LOG(INFO,"Butterworth low pass time: %f", static_cast<double>(t2 - t1) /
  CLOCKS_PER_SEC); butter_low_img =
  butter_low_img(cv::Range(0,gray_img.rows),cv::Range(0,gray_img.cols));

  // show
  cv::imshow("butter_low_img",
  ConvertSingleChannelMat2Uint8Mat<float>(butter_low_img));
*/
  // 创建高斯滤波器
  ImageGaussianFilter(gray_img, idft_img, 2.6);
  cv::imshow("ifft", idft_img / 255.0);

  cv::waitKey(0);
  return 0;
}