#include "sift.h"
namespace MY_IMG {
void SIFT(Image &img, SiftParam param,std::vector<SiftPointDescriptor> &descriptors){
    // 初始化图像金字塔
    img.Octaves.resize(param.octave_num);
    for (int i = 0; i < param.octave_num; ++i) {
        img.Octaves[i].layers.resize(param.octave_layer_num);
        img.Octaves[i].dog_layers.resize(param.octave_layer_num);
    }
}

} // namespace MY_IMG