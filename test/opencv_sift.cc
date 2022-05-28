#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "image_operator.h"

using namespace std;
using namespace cv;

int main(int argc,char **argv) {
  if (argc != 2) {
    cout << "Usage : opencv_sift image_path1 image_path2" << endl;
    return 0;
  }
  Mat image = imread(argv[1],IMREAD_COLOR);
  
  Mat gray;
  cvtColor(image, gray, COLOR_BGR2GRAY);
  vector<KeyPoint> keypoints;
  Mat descriptorsFloat, descriptorsUchar;
  Ptr<SIFT> siftFloat = cv::SIFT::create(0, 3, 0.04, 10, 1.6, CV_32F);
  siftFloat->detectAndCompute(gray, Mat(), keypoints, descriptorsFloat, false);

  Mat img_feature;
  MY_IMG::DrayPoints(image,keypoints,img_feature);
  imshow("img_feature",img_feature);
  waitKey(0);

  return 0;
}