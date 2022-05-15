#include "match.h"

namespace MY_IMG {
using Tree = KDTree<std::shared_ptr<KeyPoint>, float>;

std::shared_ptr<Tree> BuildKDTree(Image &src_featured) {
  std::shared_ptr<Tree> kp_tree(
      new Tree(src_featured.keypoints, src_featured.keypoints[0]->descriptor.size()));
  kp_tree->Build();
  return kp_tree;
}

void Match(const std::shared_ptr<Tree> &tree,
           const std::vector<std::shared_ptr<KeyPoint>> &kps,
           std::vector<std::pair<std::shared_ptr<KeyPoint>,
                                 std::vector<std::shared_ptr<KeyPoint>>>>
               &match_result,
           int k) {
  for (auto &kp : kps) {
    std::vector<std::shared_ptr<MY_IMG::KeyPoint>> result;
    tree->Query(kp, k, result);
    double dist = 0;
    for (int i = 0; i < result.size(); i++) {  
      for (int j = 0; j < kp->descriptor.size(); j++) {
        dist += sqr(kp->descriptor[j] - result[i]->descriptor[j]);
      }
    }
    LOG("dist : %f", dist);
    if (dist > MAX_MATCH_DISTANCE) {
      continue;
    }
    match_result.push_back(std::make_pair(kp, result));
  }
}
} // namespace MY_IMG