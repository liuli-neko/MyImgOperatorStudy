#include <iostream>
#include <pthread.h>

using namespace std;

int main(int argc, char **argv)
{
    int n = 10;
    cin >> n;
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < n; i++)
    {
        cout << i << endl;
    }
    return 0;
}