#pragma once
 
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<vector>
 
using namespace cv;
using namespace std;
 
 
class myDisplay
{
public:
	myDisplay() {}
	void mosaic_pyramid(const vector<vector<Mat>>& pyramid, Mat& pyramid_image, int nOctaceLayers, string str);
 
	void write_mosaic_pyramid(const vector<vector<Mat>>& gauss_pyr_1, const vector<vector<Mat>>& dog_pyr_1,
		const vector<vector<Mat>>& gauss_pyr_2, const vector<vector<Mat>>& dog_pyr_2, int nOctaveLayers);
 
	//没用到，后文没有对其进行定义
	void write_keys_image(vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2,
		const Mat& image_1, const Mat& image_2, Mat& image_1_keys, Mat& image_2_keys, bool double_size);
};