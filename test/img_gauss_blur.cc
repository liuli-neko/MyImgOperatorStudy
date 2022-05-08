#include "unit.h"
#include <ctime>
#include <iostream>

using namespace MY_IMG;

int main(int argc, char **argv) {

  std::string file_name = "./test_img.png";
  std::string output_name = "./test_img_out.png";
  double sigma = 2.0;
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
  cv::Mat blur_img;
  clock_t t1, t2;
  t1 = clock();
  GaussBlur(img, blur_img, sigma);
  t2 = clock();
  LOG("ImgFilter time: %f", static_cast<double>(t2 - t1) / CLOCKS_PER_SEC);
  cv::imshow("blur_img", blur_img);
  cv::imwrite(output_name, blur_img);

  cv::waitKey(0);
  return 0;
}