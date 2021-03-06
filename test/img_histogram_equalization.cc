#include "unit.h"
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
  // first get gray img
  cv::Mat gray_img;
  time_t t1 = clock();
  Rgb2Gray(img, gray_img);
  time_t t2 = clock();
  LOG(INFO,"rgb2gray time: %f", static_cast<double>(t2 - t1) / CLOCKS_PER_SEC);
  cv::imshow("gray",gray_img);
  // then get histogram
  cv::Mat hist_img;
  t1 = clock();
  HistogramEqualization(gray_img, hist_img);
  t2 = clock();
  LOG(INFO,"HistogramEqualization time: %f", static_cast<double>(t2 - t1) / CLOCKS_PER_SEC);
  // show
  cv::imshow("hist_img", hist_img);

  cv::waitKey(0);
  return 0;
}