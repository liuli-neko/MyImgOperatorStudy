#include "image_operator.h"
#include "match.h"
#include "sift.h"
#include <time.h>

int main(int argc, char const *argv[]) {
  std::string img1_path = "test.png", img2_path = "test2.png";
  if (argc > 2) {
    img1_path = argv[1];
    img2_path = argv[2];
  } else {
    ASSERT(argc == 2, "Usage: ./img_sift img1 img2");
  }
  IMG_Mat img1 = cv::imread(img1_path);
  IMG_Mat img2 = cv::imread(img2_path);
  MY_IMG::Image src_img1, src_img2;
  src_img1.img = img1;
  src_img1.imgId = 1;
  src_img2.img = img2;
  src_img2.imgId = 2;
  time_t start, end;
  time(&start);
  MY_IMG::SIFT(src_img1);
  time(&end);
  LOG("SIFT use time: %f", difftime(end, start));
  time(&start);
  MY_IMG::SIFT(src_img2);
  time(&end);
  LOG("SIFT use time: %f", difftime(end, start));

  std::shared_ptr<MY_IMG::Tree> tree1 = MY_IMG::BuildKDTree(src_img1);
  std::shared_ptr<MY_IMG::Tree> tree2 = MY_IMG::BuildKDTree(src_img2);

  std::vector<std::pair<std::shared_ptr<MY_IMG::KeyPoint>,
                        std::shared_ptr<MY_IMG::KeyPoint>>>
      match_result;
  MY_IMG::Match(tree1, tree2, match_result);
  for (auto &kp : match_result) {
    LOG("(%d %d)-(%d %d)", kp.first->x, kp.first->y, kp.second->x, kp.second->y);
  }

  IMG_Mat src_out1,src_out2;
  MY_IMG::DrawPoints(src_img1.img,src_img1.keypoints,src_out1);
  MY_IMG::DrawPoints(src_img2.img,src_img2.keypoints,src_out2);

  cv::imwrite("./image1_feature.png", src_out1);
  cv::imwrite("./image2_feature.png", src_out2);

  IMG_Mat match_out;
  MY_IMG::DrawMatch(src_img1.img, src_img2.img, match_result, match_out);

  cv::imwrite("./match.png", match_out);
  return 0;
}