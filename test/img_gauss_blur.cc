#include "unit.h"
#include <ctime>
#include <iostream>


using namespace MY_IMG;

int main(int argc, char **argv) {

  std::string file_name = "./test_img.png";
  std::string output_name = "./test_img_out.png";
  double sigma = 1.0;
  if (argc >= 2) {
    file_name = std::string(argv[1]);
    if (argc >= 3) {
      output_name = std::string(argv[2]);
    }
    if (argc >= 4) {
      sigma = std::stod(argv[3]);
    }
  }

  cv::Mat img = cv::imread(file_name, cv::IMREAD_COLOR);
  cv::Mat blur_filter;
  time_t t1 = clock();
  CreateGaussBlurFilter(blur_filter, sigma, -1, -1);
  time_t t2 = clock();
  LOG("CreateGaussBlurFilter time: %f", static_cast<double>(t2 - t1) / CLOCKS_PER_SEC);
  LOG("blur_filter size: [%d,%d]", blur_filter.rows, blur_filter.cols);
  cv::Mat blur_img;
  t1 = clock();
  MY_IMG::ImgFilter(img, blur_filter, blur_img);
  t2 = clock();
  LOG("ImgFilter time: %f", static_cast<double>(t2 - t1) / CLOCKS_PER_SEC);
  cv::imshow("blur_img", blur_img);
  cv::imwrite(output_name, blur_img);

  cv::waitKey(0);
  return 0;
}