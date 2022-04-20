#include <iostream>
#include "unit.h"

int main(int argc,char **argv){

    std::string img_path =  std::string(argv[1]);
    std::string output = "morphology_test.png";

    cv::Mat src = cv::imread(img_path,cv::IMREAD_COLOR);
    cv::Mat gray;
    MY_IMG::Rgb2Gray(src,gray);
    cv::Mat struct_element = (cv::Mat_<int>(3,3) << 0,64,0,
                                                   64,64,64,
                                                   0,64,0);

    cv::Mat dst_open = MY_IMG::GrayOpening(gray,struct_element);
    cv::Mat dst_close = MY_IMG::GrayClosing(gray,struct_element);

    cv::imshow("gray",gray);
    cv::imshow("opening",dst_open);
    cv::imshow("closing",dst_close);

    cv::waitKey(0);

    return 0;
}