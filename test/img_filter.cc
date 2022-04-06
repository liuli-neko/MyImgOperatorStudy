#include "unit.h"
#include <iostream>


using namespace MY_IMG;

int main(int argc, char **argv) {
  std::string file_name = "./test_img.png";
  if (argc >= 2) {
    file_name = std::string(argv[1]);
  }
  cv::Mat img = cv::imread(file_name, cv::IMREAD_COLOR);
  cv::Mat filter_img;
  ImgFilter(img, H1, filter_img);
  cv::imshow("H1", filter_img);
  ImgFilter(img, H2, filter_img);
  cv::imshow("H2", filter_img);
  ImgFilter(img, H3, filter_img);
  cv::imshow("H3", filter_img);
  ImgFilter(img, H4, filter_img);
  cv::imshow("H4", filter_img);
  ImgFilter(img, H5, filter_img);
  cv::imshow("H5", filter_img);
  ImgFilter(img, H6, filter_img);
  cv::imshow("H6", filter_img);
  ImgFilter(img, H7, filter_img);
  cv::imshow("H7", filter_img);
  ImgFilter(img, H8, filter_img);
  cv::imshow("H8", filter_img);
  ImgFilter(img, H9, filter_img);
  cv::imshow("H9", filter_img);

  cv::waitKey(0);
  return 0;
}
