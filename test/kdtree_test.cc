#include "kdtree.h"

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;
struct Point {
  int x,y;
  std::vector<float> descriptor;
  Point() {}
  float operator[](int i) const { return descriptor.at(i); }
};
int main() {
  vector<std::shared_ptr<Point>> points_1, points_2;

  int n;
  float x, y;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> x >> y;
    auto p = std::make_shared<Point>();
    p->descriptor.push_back(x);
    p->descriptor.push_back(y);
    p->x = x;
    p->y = y;
    points_1.push_back(p);
  }

  MY_IMG::KDTree<std::shared_ptr<Point>,float> tree(points_1, points_1[0]->descriptor.size());
  tree.Build();
  int q,k;
  cin >> q >> k;
  for (int i = 0; i < q; i++) {
    cin >> x >> y;
    auto p = std::make_shared<Point>();
    p->descriptor.push_back(x);
    p->descriptor.push_back(y);
    p->x = x;
    p->y = y;
    tree.Query(p, k, points_2);
    cout << "----------------------------------" << endl;
    for (auto &p : points_2) {
      cout << p->descriptor[0] << " " << p->descriptor[1] << endl;
    }
    points_2.clear();
    cout << "----------------------------------" << endl;
  }

  return 0;
}