#include "image_operator.h"
#include "sift.h"
#include <time.h>

int main(int argc, char const *argv[]) {
  std::string img1_path = "test.png", img2_path = "test2.png";
  if (argc > 1) {
    img1_path = argv[1];
  } else {
    ASSERT(argc == 2, "Usage: ./img_detection_keypoint img1");
  }
  IMG_Mat img1 = cv::imread(img1_path);
  MY_IMG::Image src_img1;
  src_img1.img = img1;
  src_img1.imgId = 1;

  MY_IMG::SIFT(src_img1);

  IMG_Mat src_out1;
  MY_IMG::DrawPoints(src_img1.img, src_img1.keypoints, src_out1);

  cv::imshow("out",src_out1);
  cv::waitKey(0);

  return 0;
}