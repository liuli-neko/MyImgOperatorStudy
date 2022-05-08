#include "image_operator.h"
#include "unit.h"
#include "all.h"

int main(int argc, char const *argv[])
{
  std::string img_path = "test.png";
  std::string output = "rota_test.png";
  double angle = 0.0;
  double zoom = 2.0;
  if(argc > 1){
    img_path = std::string(argv[1]);
  }
  if(argc > 2){
    output = std::string(argv[2]);
  }
  if(argc > 3){
    zoom = std::stod(argv[3]);
  }
  if(argc > 4){
    angle = std::stod(argv[4]);
  }
  cv::Mat src;
  MY_IMG::Rgb2Gray(cv::imread(img_path), src, {1/3.0, 1/3.0, 1/3.0});
  cv::Mat dst;
  MY_IMG::ImageChange(src, dst, zoom, angle);
  LOG("zoom:%f,angle:%f",zoom,angle);
  cv::imshow("dst", dst);
  cv::waitKey(0);
  cv::imwrite(output, dst);
  return 0;
}
