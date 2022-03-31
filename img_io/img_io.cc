#include "img_io.h"

namespace MY_IMG
{
    ImageIO::ImageIO()
    {
    }
    ImageIO::ImageIO(const std::string &file_path)
    {
        read(file_path);
    }
    ImageIO::ImageIO(const cv::Mat &img)
    {
        img_ = img;
    }
    cv::Mat ImageIO::GetData(){
        return img_;
    }
    void ImageIO::read(const std::string &file_path)
    {
        img_ = cv::imread(file_path, cv::IMREAD_COLOR);
    }
    void ImageIO::show(const std::string &title)
    {
        cv::imshow(title, img_);
    }
    void ImageIO::save(const std::string &file_path)
    {
        cv::imwrite(file_path, img_);
    }
    void ImageIO::rgb2gray(cv::Mat &target_img) {
        int width = img_.cols;
        int height = img_.rows;
        target_img = cv::Mat(img_.size(),CV_8UC1);

        for (int i = 0;i<height;i++){
            for(int j = 0;j < width;j++){
                const cv::Vec3b &pixel = img_.at<cv::Vec3b>(i,j);

                target_img.at<uint8_t>(i,j) = 0.2989*pixel[0] + 0.5870*pixel[1] + 0.1140*pixel[2];
            }
        }
    }
}
