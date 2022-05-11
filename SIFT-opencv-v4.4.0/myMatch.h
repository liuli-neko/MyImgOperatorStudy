#pragma once
 
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
 
#include<vector>
#include<string>
#include<iostream>
 
const double dis_ratio1 = 0.75;			//最近邻和次近邻距离比阈值，就目前测试来看 dis_ratio = 0.75 时正确匹配的数量相对较多
const double dis_ratio2 = 0.8;
const double dis_ratio3 = 0.9;
 
const float ransac_error = 1.5;			//ransac算法误差阈值
 
const double FSC_ratio_low = 0.8;
 
const double FSC_ratio_up = 1;
 
const int pointsCount = 9;				// k均值聚类数据点个数
const int clusterCount = 3;				// k均值聚类质心的个数
 
 
enum DIS_CRIT { Euclidean = 0, COS };	//距离度量准则
 
using namespace std;
using namespace cv;
 
 
class myMatch
{
public:
 
	myMatch();
	~myMatch();
 
	//该函数根据正确的匹配点对，计算出图像之间的变换关系
	Mat LMS(const Mat& match1_xy, const Mat& match2_xy, string model, float& rmse);
	
	//改进版LMS超定方程
	Mat improve_LMS(const Mat& match1_xy, const Mat& match2_xy, string model, float& rmse);
 
	//该函数删除错误的匹配点对
	Mat ransac(const vector<Point2f>& points_1, const vector<Point2f>& points_2, string model, float threshold, vector<bool>& inliers, float& rmse);
 
	//绘制棋盘图像
	void mosaic_map(const Mat& image_1, const Mat& image_2, Mat& chessboard_1, Mat& chessboard_2, Mat& mosaic_image, int width);
 
	//该函数把两幅配准后的图像进行融合镶嵌
	void image_fusion(const Mat& image_1, const Mat& image_2, const Mat T, Mat& fusion_image, Mat& matched_image);
 
	//该函数进行描述子的最近邻和次近邻匹配
	void match_des(const Mat& des_1, const Mat& des_2, vector<vector<DMatch>>& dmatchs, DIS_CRIT dis_crite);
 
	//建立尺度直方图、ROM 直方图
	void scale_ROM_Histogram(const vector<DMatch>& matches, float* scale_hist, float* ROM_hist, int n);
 
	//该函数删除错误匹配点对，并进行配准
	Mat match(const Mat& image_1, const Mat& image_2, const vector<vector<DMatch>>& dmatchs, vector<KeyPoint> keys_1,
		vector<KeyPoint> keys_2, string model, vector<DMatch>& right_matchs, Mat& matched_line, vector<DMatch>& init_matchs);
};
 