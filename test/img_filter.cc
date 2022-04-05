#include <iostream>
#include "img_io.h"
#include "unit.h"

using namespace MY_IMG;

int main(int argc, char **argv)
{
    std::string file_name = "./test_img.png";
    if (argc >= 2)
    {
        file_name = std::string(argv[1]);
    }
    ImageIO img(file_name);
    cv::Mat filter_img;
    ImgFilter(img.GetData(), H1, filter_img);
    cv::imshow("H1", filter_img);
    ImgFilter(img.GetData(), H2, filter_img);
    cv::imshow("H2", filter_img);
    ImgFilter(img.GetData(), H3, filter_img);
    cv::imshow("H3", filter_img);
    ImgFilter(img.GetData(), H4, filter_img);
    cv::imshow("H4", filter_img);
    ImgFilter(img.GetData(), H5, filter_img);
    cv::imshow("H5", filter_img);
    ImgFilter(img.GetData(), H6, filter_img);
    cv::imshow("H6", filter_img);
    ImgFilter(img.GetData(), H7, filter_img);
    cv::imshow("H7", filter_img);
    ImgFilter(img.GetData(), H8, filter_img);
    cv::imshow("H8", filter_img);
    ImgFilter(img.GetData(), H9, filter_img);
    cv::imshow("H9", filter_img);

    cv::waitKey(0);
    return 0;
}
