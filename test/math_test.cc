#include <cmath>
#include <complex>
#include <iostream>

using namespace std;

const double PI = acos(-1);

int main(){

    // 声明一个复数
    complex<double> c(1.0, 2.0);
    
    // 遍历单位圆上的所有复数点
    for (int i = 0; i < 100; i++) {
        // 计算复数的角度
        double b = -2 * M_PI * i / 100;
        c = complex<double>(0, b);
        // 输出结果
        cout << "b = " << b << endl;
        cout << "exp(c): " << exp(c) << endl;
    }


    return 0;
}