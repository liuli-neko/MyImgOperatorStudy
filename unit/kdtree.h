#ifndef IMG_OPERATOR_UNIT_KDTREE_H_
#define IMG_OPERATOR_UNIT_KDTREE_H_
#include <algorithm>
#include <iostream>
#include <memory>
#include <queue>
#include <vector>

#include "all.h"

namespace MY_IMG {

template <typename Element, typename val_type = int> struct KDTreeNode {
  Element element;
  int left;
  int right;
  std::vector<val_type> max_val;
  std::vector<val_type> min_val;
  KDTreeNode() : left(-1), right(-1) {}
};
template <typename Element, typename val_type = int> class KDTree {
public:
  /**
   * @brief 构造函数
   * @param elements 元素集合
   * @param dim 元素维度
   */
  std::vector<Element> &elements;
  KDTree(std::vector<Element> &elements, int dim) : elements(elements), n_(dim) {
    LOG(INFO,"create kdtree...");
    LOG(INFO,"dim: %d", dim);
    nodes_.resize(elements.size());
    for (int i = 0; i < elements.size(); i++) {
      nodes_.at(i) = std::shared_ptr<KDTreeNode<Element, val_type>>(
          new KDTreeNode<Element, val_type>());
      nodes_.at(i)->element = elements[i];
    }
  }
  ~KDTree() { nodes_.clear(); }
  void Build() {
    int root = build(0, nodes_.size() - 1, 0);
    LOG(INFO,"the root : %d,size : %ld", root, nodes_.size());
  }
  void Query(const Element &element, int k, std::vector<Element> &result) {
    while (q_.size()) {
      q_.pop();
    }
    for (int i = 0; i < (k << 1); i++) {
      q_.push(std::make_pair(1e9, 1e18));
    }
    query(0, nodes_.size() - 1, element);
    while (q_.size() > k) {
      q_.pop();
    }
    for (int i = 0; i < k; i++) {
      if (q_.top().first == 1e9) {
        q_.pop();
        continue;
      }
      ASSERT(q_.top().first < nodes_.size(), "q_.top.first[%d] < nodes_.size()[%ld]", q_.top().first,
             nodes_.size());
      LOG(INFO,"q : [%d] [%lf]", q_.top().first, q_.top().second);
      result.push_back(nodes_.at(q_.top().first)->element);
      q_.pop();
    }
    std::reverse(result.begin(), result.end());
  }

private:
  void update(int u) {
    nodes_.at(u)->max_val.resize(n_);
    nodes_.at(u)->min_val.resize(n_);
    for (int i = 0; i < n_; i++) {
      nodes_.at(u)->max_val.at(i) = nodes_.at(u)->min_val.at(i) =
          (*(nodes_.at(u)->element))[i];
    }
    if (nodes_.at(u)->left != -1) {
      for (int i = 0; i < n_; i++) {
        nodes_.at(u)->max_val.at(i) = std::max(nodes_.at(u)->max_val.at(i),
                                         nodes_.at(nodes_.at(u)->left)->max_val.at(i));
        nodes_.at(u)->min_val.at(i) = std::min(nodes_.at(u)->min_val.at(i),
                                         nodes_.at(nodes_.at(u)->left)->min_val.at(i));
      }
    }
    if (nodes_.at(u)->right != -1) {
      for (int i = 0; i < n_; i++) {
        nodes_.at(u)->max_val.at(i) = std::max(nodes_.at(u)->max_val.at(i),
                                         nodes_.at(nodes_.at(u)->right)->max_val.at(i));
        nodes_.at(u)->min_val.at(i) = std::min(nodes_.at(u)->min_val.at(i),
                                         nodes_.at(nodes_.at(u)->right)->min_val.at(i));
      }
    }
  }

  int build(int l, int r, int flag) {
    if (l > r) {
      return -1;
    }

    int mid = (l + r) >> 1;

    std::nth_element(
        nodes_.begin() + l, nodes_.begin() + mid, nodes_.begin() + r + 1,
        [&flag](const std::shared_ptr<KDTreeNode<Element, val_type>> &a,
                const std::shared_ptr<KDTreeNode<Element, val_type>> &b)
            -> bool { return (*(a->element))[flag] < (*(b->element))[flag]; });

    nodes_.at(mid)->left = build(l, mid - 1, (flag + 1) % n_);
    nodes_.at(mid)->right = build(mid + 1, r, (flag + 1) % n_);

    update(mid);
    return mid;
  }

  double getMax(int x, const Element &elem) {
    double dist = 0;
    if (x <= -1) {
      return -1;
    }
    ASSERT(x >= 0 && x < nodes_.size(), "index error : %d > nodes_.size(%ld)", x, nodes_.size());
    for (int i = 0; i < n_; i++) {
      dist += std::max(sqr(static_cast<double>(
                           (*elem)[i] - nodes_.at(x)->max_val.at(i))),
                       sqr(static_cast<double>(
                           (*elem)[i] - nodes_.at(x)->min_val.at(i))));
    }
    return dist;
  }

  double getMin(int x,const Element &elem) {
    double dist = 0;
    if (x <= -1) {
      return 1e9;
    }
    ASSERT(x >= 0 && x < nodes_.size(), "index error : %d > nodes_.size(%ld)", x, nodes_.size());
    for (int i = 0; i < n_; i++) {
      dist += std::min(sqr(static_cast<double>(
                           (*elem)[i] - nodes_.at(x)->max_val.at(i))),
                       sqr(static_cast<double>(
                           (*elem)[i] - nodes_.at(x)->min_val.at(i))));
    }
    return dist;
  }

  void query(int l, int r, const Element &elem) {
    if (l > r){
      return;
    }
    ASSERT(l >= 0 && l < nodes_.size(), "index error : %d > nodes_.size(%ld)", l, nodes_.size());
    ASSERT(r >= 0 && r < nodes_.size(), "index error : %d > nodes_.size(%ld)", r, nodes_.size());
    int mid = (l + r) >> 1;
    double dist = 0;
    for (int i = 0; i < n_; i++) {
      dist += sqr(static_cast<double>((*elem)[i] - (*(nodes_.at(mid)->element))[i]));
    }
    if (dist <= q_.top().second) {
      q_.pop();
      q_.push({mid, dist});
    }
    // LOG(INFO,"l: %d, r: %d, mid : %d, dist: %lf", l, r, mid, dist);
    double distl = getMin(nodes_.at(mid)->left, elem);
    double distr = getMin(nodes_.at(mid)->right, elem);
    // LOG(INFO,"distl: %lf, distr: %lf", distl, distr);
    if (distl < distr) {
      if (distl <= q_.top().second) {
        query(l, mid - 1, elem);
      }
      if (distr <= q_.top().second) {
        query(mid + 1, r, elem);
      }
    } else {
      if (distr <= q_.top().second) {
        query(mid + 1, r, elem);
      }
      if (distl <= q_.top().second) {
        query(l, mid - 1, elem);
      }
    }
  }
  struct cmp {
    bool operator()(const std::pair<int, double> &a,
                    const std::pair<int, double> &b) {
      if (a.second == b.second) {
        return a.first < b.first;
      }
      return a.second < b.second;
    }
  };
  std::priority_queue<std::pair<int, double>,
                      std::vector<std::pair<int, double>>, cmp>
      q_;
  std::vector<std::shared_ptr<KDTreeNode<Element, val_type>>> nodes_;
  int n_;
};
} // namespace MY_IMG
#endif