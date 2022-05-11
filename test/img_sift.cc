#include "sift.h"
#include "image_operator.h"

int main(int argc, char const *argv[])
{
  std::string img_path = "test.png";
  if(argc > 0) img_path = argv[1];
  IMG_Mat src = cv::imread(img_path);
  MY_IMG::Image img;
  img.img = src;
  img.imgId = 0;
  std::vector<MY_IMG::KeyPoint> descriptors;
  MY_IMG::SIFT(img, descriptors);

  return 0;
}