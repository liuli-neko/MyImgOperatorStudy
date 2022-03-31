#ifndef IMG_IO_H_
#define IMG_IO_H_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>

namespace MY_IMG
{
    class ImageIO
    {
    public:
        ImageIO();
        explicit ImageIO(const std::string &file_path);
        void read(const std::string &file_path);
        void show(const std::string &title = "img");
        void save(const std::string &file_path);
        void rgb2gray(cv::Mat &target_img);
        cv::Mat GetData();
    private:
        cv::Mat img_;
    };
}

#endif