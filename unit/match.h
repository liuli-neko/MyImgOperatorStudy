#ifndef IMG_OPERATOR_UNIT_MATCH_H_
#define IMG_OPERATOR_UNIT_MATCH_H_

#include "all.h"
#include "kdtree.h"
#include <memory>

namespace MY_IMG {
#define MAX_MATCH_DISTANCE 0.005
using Tree = KDTree<std::shared_ptr<KeyPoint>, float>;

std::shared_ptr<Tree> BuildKDTree(Image &src_featured);

void Match(const std::shared_ptr<Tree> &tree,
           const std::vector<std::shared_ptr<KeyPoint>> &kps,
           std::vector<std::pair<std::shared_ptr<KeyPoint>,
                                 std::vector<std::shared_ptr<KeyPoint>>>>
               &match_result,
           int k);

} // namespace MY_IMG

#endif