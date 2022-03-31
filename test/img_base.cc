#include <iostream>
#include "img_io.h"

std::string file_name = "./test_img.png";
std::string output_name = "./test_img_out.png";

int main(int argc, char **argv)
{

    if (argc >= 2)
    {
        file_name = std::string(argv[1]);
        if (argc >= 3)
        {
            output_name = std::string(argv[2]);
        }
    }
    MY_IMG::ImageIO img(file_name);
    img.show();
    img.save(output_name);

    cv::waitKey(0);
    return 0;
}