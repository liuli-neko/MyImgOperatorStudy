#ifndef IMG_OPERATOR_UNIT_MATCH_H_
#define IMG_OPERATOR_UNIT_MATCH_H_

#include "all.h"
#include "kdtree.h"
#include <memory>

namespace MY_IMG {
#define MAX_MATCH_DISTANCE 0.00005
#define MAX_MATCH_BLOCK_SIZE 10
using Tree = KDTree<std::shared_ptr<KeyPoint>, float>;

std::shared_ptr<Tree> BuildKDTree(Image &src_featured);

void Match(const std::shared_ptr<Tree> &kps1,
           const std::shared_ptr<Tree> &kps2,
           std::vector<std::pair<std::shared_ptr<KeyPoint>,
                                 std::shared_ptr<KeyPoint>>>
               &match_result);

void Match(const std::shared_ptr<Tree> &kps1,
           const std::shared_ptr<Tree> &kps2,
           std::vector<std::pair<std::shared_ptr<KeyPoint>,
                                 std::shared_ptr<KeyPoint>>>
               &match_result,bool ergodic);

} // namespace MY_IMG

#endif