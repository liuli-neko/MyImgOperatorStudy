#include "img_io.h"
#include "unit.h"
#include <fstream>
#include <iostream>
#include <vector>

using MY_IMG::CreateGaussBlurFilter;
using MY_IMG::HistogramEqualization;
using MY_IMG::ImageIO;

std::string file_name = "./test_img.png";

int main(int argc, char **argv)
{
  double sigma = 1;
  if (argc >= 2)
  {
    file_name = std::string(argv[1]);
    if (argc >= 3)
    {
      sigma = std::stod(argv[2]);
    }
  }
  std::ifstream file(file_name, std::ios::binary);
  if (!file.is_open())
  {
    printf("file open faild\n");
    return 0;
  }
  file.close();
  // read img and show
  ImageIO img(file_name);
  // img.show();

  // make color image to gray and show
  cv::Mat gray_img;
  img.rgb2gray(gray_img);
  cv::imshow("gray_img", gray_img);
  // cv::imwrite("e:/workplace/C++/img_operator/gray_img.png",gray_img);

  // do histogram equalization and show
  cv::Mat dimg;
  // HistogramEqualization(gray_img, dimg);
  // cv::imshow("his", dimg);
  // cv::imwrite("e:/workplace/C++/img_operator/his.png",dimg);

  // cv::Mat H;
  // double h[3][3] = {{-2, -1, 0}
  //                 , {-1, 0, 1}
  //                 , {0, 1, 2}};
  // H = cv::Mat(3, 3, CV_64F, h);
  // cv::Mat H_img;
  // MY_IMG::ImgFilter(gray_img, H, H_img, true);
  // cv::imshow("H_img", H_img);
  // 统计下面代码的运行时间
  // 获取当前时间
  clock_t start, end;
  start = clock();
  MY_IMG::DFT(gray_img, dimg);
  cv::imshow("dft",dimg);
  end = clock();
  LOG("time:%f", (double)(end - start) / CLOCKS_PER_SEC);
  // cv::imwrite("/mnt/windows/data/workplace/C++/img_operator/dft.png",dimg);

  // create laplacian filter and show
  // H1=[[0,1,0],[1,-4,1],[0,1,0]]
  // H2=[[1,1,1],[1,-8,1],[1,1,1]]
  // H3=[[1,-2,1],[-2,4,-2],[1,-2,1]]
  // H4=[[0,-1,0],[-1,4,-1],[0,-1,0]]
  // H5=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
  // 生成以上五个滤波器

  // cv::Mat H1, H2, H3, H4, H5, H6, H7, H8;
  // double h1[3][3] = {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};
  // double h2[3][3] = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
  // double h3[3][3] = {{1, -2, 1}, {-2, 4, -2}, {1, -2, 1}};
  // double h4[3][3] = {{0, -1, 0}, {-1, 4, -1}, {0, -1, 0}};
  // double h5[3][3] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
  // double h6[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
  // double h7[3][3] = {{-1, -1, -1}, {-1, 9, -1}, {-1, -1, -1}};
  // double h8[3][3] = {{0, 1, 0}, {1, -5, 1}, {0, 1, 0}};
  // H1 = cv::Mat(3, 3, CV_64F, h1);
  // H2 = cv::Mat(3, 3, CV_64F, h2);
  // H3 = cv::Mat(3, 3, CV_64F, h3);
  // H4 = cv::Mat(3, 3, CV_64F, h4);
  // H5 = cv::Mat(3, 3, CV_64F, h5);
  // H6 = cv::Mat(3, 3, CV_64F, h6);
  // H7 = cv::Mat(3, 3, CV_64F, h7);
  // H8 = cv::Mat(3, 3, CV_64F, h8);
  // cv::Mat laplacian_img;
  // MY_IMG::ImgFilter(img.GetData(), H1, laplacian_img, true);
  // cv::imshow("H1", laplacian_img);
  // MY_IMG::ImgFilter(img.GetData(), H2, laplacian_img, true);
  // cv::imshow("H2", laplacian_img);
  // MY_IMG::ImgFilter(gray_img, H3, laplacian_img, true);
  // cv::imshow("H3",laplacian_img);
  // MY_IMG::ImgFilter(gray_img, H4, laplacian_img,true);
  // cv::imshow("H4",laplacian_img);
  // MY_IMG::ImgFilter(gray_img, H5, laplacian_img,true);
  // cv::imshow("H5",laplacian_img);
  // MY_IMG::ImgFilter(img.GetData(), H6, laplacian_img);
  // cv::imshow("H6", laplacian_img);
  // MY_IMG::ImgFilter(img.GetData(), H7, laplacian_img);
  // cv::imshow("H7", laplacian_img);
  // MY_IMG::ImgFilter(gray_img, H8, laplacian_img);
  // cv::imshow("H8",laplacian_img);

  // create gauss blur filter and show
  /*
    cv::Mat blur_img, blur_img1;

    cv::Mat blur_filter, blur_filter1;

    CreateGaussBlurFilter(blur_filter, sigma, -1, -1);

    LOG("blur_filter :[%d,%d]", blur_filter.size().height,
        blur_filter.size().width);
    LOG("blur_filter_sum : %f", cv::sum(blur_filter)[0]);

    MY_IMG::ImgFilter(img.GetData(), blur_filter, blur_img);

    cv::imshow("blur_filter", blur_img);
  */
  // 当图像窗口被关闭时，程序退出
  cv::waitKey(0);

  return 0;
}