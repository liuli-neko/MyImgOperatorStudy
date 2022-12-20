#include <opencv2/opencv.hpp>

#include <iostream>
#include <type_traits>

using namespace std;
using namespace cv;

int main() {

  Mat img = Mat(3, 7, CV_8U);

  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      img.at<uint8_t>(i, j) = i * img.cols + j;
    }
  }

  cout << img << endl << endl;

  cout << format(img, Formatter::FMT_PYTHON) << endl << endl;

  return 0;
}