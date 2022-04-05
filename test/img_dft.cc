#include <iostream>
#include <ctime>
#include "img_io.h"
#include "unit.h"

using namespace MY_IMG;

int main(int argc, char **argv)
{
    std::string file_name = "./test_img.png";
    std::string output_name = "./test_img_out.png";
    if (argc >= 2)
    {
        file_name = std::string(argv[1]);
        if (argc >= 3)
        {
            output_name = std::string(argv[2]);
        }
    }
    ImageIO img(file_name);
    // get dft
    cv::Mat dft_img;
    cv::Mat gray_img;
    img.rgb2gray(gray_img);
    time_t t1 = clock();
    DFT(gray_img, dft_img);
    time_t t2 = clock();
    LOG("DFT time: %f", static_cast<double>(t2 - t1) / 1000.0);
    // show
    cv::imshow("dft_img",ConvertComplexMat2doubleMat(dft_img));

    // 创建一个理想低通滤波器
    cv::Mat butter_filter = cv::Mat::zeros(gray_img.size(), CV_64FC1);
    auto Dist = [x0 = gray_img.rows/2,y0 = gray_img.cols/2](int x, int y) {
        return sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0));
    };
    // 设置截止频率和阶数
    double D0 = 10;
    int n = 2;
    auto Butterworth = [Dist, &n, &D0](int x, int y) {
        return 1.0 / (1.0 + pow(Dist(x, y) / D0, 2 * n));
    };
    // 进行滤波
    for (int i = 0;i<gray_img.rows;i++)
    {
        for (int j = 0;j<gray_img.cols;j++)
        {
            dft_img.at<std::complex<double>>(i, j) *= Butterworth(i, j);
        }
    }
    // get idft
    cv::Mat idft_img;
    t1 = clock();
    IDFT(dft_img, idft_img);
    t2 = clock();
    LOG("IDFT time: %f", static_cast<double>(t2 - t1) / 1000.0);
    // show
    cv::imshow("idft_img", idft_img);

    cv::waitKey(0);
    return 0;
}