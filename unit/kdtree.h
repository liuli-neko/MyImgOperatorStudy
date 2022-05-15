#ifndef IMG_OPERATOR_UNIT_KDTREE_H_
#define IMG_OPERATOR_UNIT_KDTREE_H_
#include <algorithm>
#include <iostream>
#include <memory>
#include <queue>
#include <vector>

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
  KDTree(std::vector<Element> &elements, int dim) : n_(dim) {
    LOG("create kdtree...");
    LOG("dim: %d", dim);
    nodes_.resize(elements.size());
    for (int i = 0; i < elements.size(); i++) {
      nodes_[i] = std::shared_ptr<KDTreeNode<Element, val_type>>(
          new KDTreeNode<Element, val_type>());
      nodes_[i]->element = elements[i];
    }
  }
  ~KDTree() { nodes_.clear(); }
  void Build() {
    int root = build(0, nodes_.size() - 1, 0);
    LOG("the root : %d,size : %ld", root, nodes_.size());
  }
  void Query(const Element &element, int k, std::vector<Element> &result) {
    while (q_.size()) {
      q_.pop();
    }
    for (int i = 0; i < (k << 1); i++) {
      q_.push(std::make_pair(-1, -1));
    }
    query(0, nodes_.size() - 1, element);
    for (int i = 0; i < k; i++) {
      if (q_.top().first == -1) {
        break;
      }
      result.push_back(nodes_[q_.top().first]->element);
      q_.pop();
    }
  }

private:
  void update(int u) {
    nodes_[u]->max_val.resize(n_);
    nodes_[u]->min_val.resize(n_);
    for (int i = 0; i < n_; i++) {
      nodes_[u]->max_val[i] = nodes_[u]->min_val[i] =
          (*(nodes_[u]->element))[i];
    }
    if (nodes_[u]->left != -1) {
      for (int i = 0; i < n_; i++) {
        nodes_[u]->max_val[i] = std::max(nodes_[u]->max_val[i],
                                         nodes_[nodes_[u]->left]->max_val[i]);
        nodes_[u]->min_val[i] = std::min(nodes_[u]->min_val[i],
                                         nodes_[nodes_[u]->left]->min_val[i]);
      }
    }
    if (nodes_[u]->right != -1) {
      for (int i = 0; i < n_; i++) {
        nodes_[u]->max_val[i] = std::max(nodes_[u]->max_val[i],
                                         nodes_[nodes_[u]->right]->max_val[i]);
        nodes_[u]->min_val[i] = std::min(nodes_[u]->min_val[i],
                                         nodes_[nodes_[u]->right]->min_val[i]);
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

    nodes_[mid]->left = build(l, mid - 1, (flag + 1) % n_);
    nodes_[mid]->right = build(mid + 1, r, (flag + 1) % n_);

    update(mid);
    return mid;
  }

  double getMax(int x, const Element &elem) {
    double dist = 0;
    if (x == -1) {
      return -1;
    }
    for (int i = 0; i < n_; i++) {
      dist += std::max(sqr(static_cast<double>(
                           (*elem)[i] - nodes_[nodes_[x]->left]->max_val[i])),
                       sqr(static_cast<double>(
                           (*elem)[i] - nodes_[nodes_[x]->right]->min_val[i])));
    }
    return dist;
  }

  void query(int l, int r, const Element &elem) {
    if (l > r)
      return;
    int mid = (l + r) >> 1;
    double dist = getMax(mid, elem);
    if (dist > q_.top().second) {
      q_.pop();
      q_.push(std::make_pair(mid, dist));
    }
    double distl = getMax(nodes_[mid]->left, elem);
    double distr = getMax(nodes_[mid]->right, elem);

    if (distl > distr) {
      if (distl > q_.top().second) {
        query(l, mid - 1, elem);
      }
      if (distr > q_.top().second) {
        query(mid + 1, r, elem);
      }
    } else {
      if (distr > q_.top().second) {
        query(mid + 1, r, elem);
      }
      if (distl > q_.top().second) {
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
      return a.second > b.second;
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