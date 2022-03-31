#include <iostream>
#include <fstream>

std::string file_name = "./test_img_out.png";

int main(int argc, char **argv)
{
    // read file by ifstream
    if (argc >= 2)
    {
        file_name = std::string(argv[1]);
    }
    std::ifstream ifs(file_name, std::ios::binary);
    if (!ifs.is_open())
    {
        std::cout << "open file failed" << std::endl;
        return -1;
    }
    std::cout << "open file success" << std::endl;

    return 0;
}