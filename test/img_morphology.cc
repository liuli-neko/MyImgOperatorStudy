#include "unit.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage : img_morphology {image_path}" << std::endl;
        exit(0);
    }
    std::string img_path = std::string(argv[1]);

    cv::Mat src  = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::Mat gray = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat struct_element = cv::Mat::ones(5, 5, CV_8UC1);
    std::cout << struct_element << std::endl;
    cv::Mat dst_erode  = MY_IMG::GrayErode(gray, struct_element);
    cv::Mat dst_dilate = MY_IMG::GrayDilate(gray, struct_element);
    cv::Mat dst_open   = MY_IMG::GrayOpening(gray, struct_element);
    cv::Mat dst_close  = MY_IMG::GrayClosing(gray, struct_element);

    cv::imwrite("gray.png", gray);                         // origin gray image
    cv::imwrite("open.png", dst_open);                     // opening operation image
    cv::imwrite("close.png", dst_close);                   // closing operation image
    cv::imwrite("erode.png", dst_erode);                   // erode operation image
    cv::imwrite("dilate.png", dst_dilate);                 // dilate operation image
    cv::imwrite("imtophat.png", dst_open - gray);          // top hat image
    cv::imwrite("imblackhat.png", dst_close - gray);       // black hat image
    cv::imwrite("imgradient.png", dst_dilate - dst_erode); // gradient image

    return 0;
}