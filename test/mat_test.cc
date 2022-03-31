#include <opencv2/opencv.hpp>

#include <iostream>
#include <type_traits>

using namespace std;
using namespace cv;

int main(int argc,char **argv){

    Mat img(5,5,CV_8UC3,Scalar(123,145,154));

    cout << img << endl << endl;
    cout << format(img,Formatter::FMT_PYTHON) << endl << endl;

    for(int i = 0 ; i<img.rows; i++){
        for(int j = 0;j < img.cols; j++){
            cout << img.at<Vec3b>(i,j) << endl;
        }
    }

    return 0;
}