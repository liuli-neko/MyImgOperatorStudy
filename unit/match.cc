#include "match.h"

namespace MY_IMG {
using Tree = KDTree<std::shared_ptr<KeyPoint>, float>;

std::shared_ptr<Tree> BuildKDTree(Image &src_featured) {
  std::shared_ptr<Tree> kp_tree(new Tree(
      src_featured.keypoints, src_featured.keypoints[0]->descriptor.size()));
  kp_tree->Build();
  return kp_tree;
}

void Match(
    const std::shared_ptr<Tree> &kps1, const std::shared_ptr<Tree> &kps2,
    std::vector<std::pair<std::shared_ptr<KeyPoint>, std::shared_ptr<KeyPoint>>>
        &match_result) {
  for (auto &kp : kps2->elements) {
    std::vector<std::shared_ptr<MY_IMG::KeyPoint>> result, result_tmp;
    std::shared_ptr<MY_IMG::KeyPoint> kp_tmp;
    kps1->Query(kp, MAX_MATCH_BLOCK_SIZE, result);
    double dist = 1e18;
    for (int i = 0; i < result.size() && dist == 1e18; i++) {
      result_tmp.clear();
      kps2->Query(result.at(i), MAX_MATCH_BLOCK_SIZE, result_tmp);
      for (auto &tmp : result_tmp) {
        if ((*tmp) == (*kp)) {
          LOG(INFO,"matched point mp(%d,%d) - mp(%d,%d)", result.at(i)->x,
              result.at(i)->y, tmp->x, tmp->y);
          dist = 0;
          kp_tmp = result.at(i);
          for (int index_d = 0; index_d < kp->descriptor.size(); index_d++) {
            dist += (kp->descriptor.at(index_d) -
                     result.at(i)->descriptor.at(index_d)) *
                    (kp->descriptor.at(index_d) -
                     result.at(i)->descriptor.at(index_d));
          }
          break;
        }
      }
    }
    if(dist != 1e18) {
      LOG(INFO,"dist : %f", dist);
    }
    if (dist > MAX_MATCH_DISTANCE) {
      continue;
    }
    match_result.push_back(std::make_pair(kp, kp_tmp));
  }
}
struct _cmp {
  bool operator()(const std::pair<double, std::shared_ptr<KeyPoint>> &a,
                  const std::pair<double, std::shared_ptr<KeyPoint>> &b) {
    return a.first < b.first;
  }
};
void _min_des_points(const std::shared_ptr<KeyPoint> &kp,
                     const std::vector<std::shared_ptr<KeyPoint>> &kps,
                     std::vector<std::shared_ptr<KeyPoint>> &result) {
  std::priority_queue<std::pair<double, std::shared_ptr<KeyPoint>>,
                      std::vector<std::pair<double, std::shared_ptr<KeyPoint>>>,
                      _cmp>
      min_des_points;
  double dist = -1;
  for (int i = 0; i < kps.size() && dist == -1; i++) {
    for (int index_d = 0; index_d < kp->descriptor.size(); index_d++) {
      dist += (kp->descriptor.at(index_d) - kps.at(i)->descriptor.at(index_d)) *
              (kp->descriptor.at(index_d) - kps.at(i)->descriptor.at(index_d));
    }
  }
  if (dist > MAX_MATCH_DISTANCE) {
    return;
  }
  if (dist < min_des_points.top().first) {
    min_des_points.pop();
    min_des_points.push(std::make_pair(dist, kp));
  }
  if (min_des_points.size() > MAX_MATCH_BLOCK_SIZE) {
    min_des_points.pop();
  }
  while (!min_des_points.empty()) {
    result.push_back(min_des_points.top().second);
    min_des_points.pop();
  }
}
void Match(
    const std::shared_ptr<Tree> &kps1, const std::shared_ptr<Tree> &kps2,
    std::vector<std::pair<std::shared_ptr<KeyPoint>, std::shared_ptr<KeyPoint>>>
        &match_result,
    bool ergodic) {
  if (!ergodic) {
    Match(kps1, kps2, match_result);
    return;
  }

  for (auto &kp : kps1->elements) {
    double dist = 0, min_dist = 1e18;
    std::shared_ptr<MY_IMG::KeyPoint> min_kp;
    for (auto &kp_tmp : kps2->elements) {
      dist = 0;
      for (int index_d = 0; index_d < kp->descriptor.size(); index_d++) {
        dist += (kp->descriptor.at(index_d) - kp_tmp->descriptor.at(index_d)) *
                (kp->descriptor.at(index_d) - kp_tmp->descriptor.at(index_d));
      }
      if (dist < min_dist) {
        min_dist = dist;
        min_kp = kp_tmp;
      }
    }

    if (min_dist > MAX_MATCH_DISTANCE) {
      continue;
    }
    LOG(INFO,"dist : %f", min_dist);
    match_result.push_back(std::make_pair(kp, min_kp));
  }
}
} // namespace MY_IMG