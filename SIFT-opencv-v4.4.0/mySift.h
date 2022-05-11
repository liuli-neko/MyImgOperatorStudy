#pragma once                    //防止头文件重复包含和下面作用一样
 
#include<iostream>
#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
 
using namespace std;
using namespace cv;
 
/*************************定义常量*****************************/
 
//高斯核大小和标准差关系，size=2*(GAUSS_KERNEL_RATIO*sigma)+1,经常设置GAUSS_KERNEL_RATIO=2-3之间
const double GAUSS_KERNEL_RATIO = 3;
 
const int MAX_OCTAVES = 8;					//金字塔最大组数
 
const float CONTR_THR = 0.04f;				//默认是的对比度阈值(D(x))
 
const float CURV_THR = 10.0f;				//关键点主曲率阈值
 
const float INIT_SIGMA = 0.5f;				//输入图像的初始尺度
 
const int IMG_BORDER = 2;					//图像边界忽略的宽度，也可以改为 1 
 
const int MAX_INTERP_STEPS = 5;				//关键点精确插值次数
 
const int ORI_HIST_BINS = 36;				//计算特征点方向直方图的BINS个数
 
const float ORI_SIG_FCTR = 1.5f;			//计算特征点主方向时候，高斯窗口的标准差因子
 
const float ORI_RADIUS = 3 * ORI_SIG_FCTR;	//计算特征点主方向时，窗口半径因子
 
const float ORI_PEAK_RATIO = 0.8f;			//计算特征点主方向时，直方图的峰值比
 
const int DESCR_WIDTH = 4;					//描述子直方图的网格大小(4x4)
 
const int DESCR_HIST_BINS = 8;				//每个网格中直方图角度方向的维度
 
const float DESCR_MAG_THR = 0.2f;			//描述子幅度阈值
 
const float DESCR_SCL_FCTR = 3.0f;			//计算描述子时，每个网格的大小因子
 
const int SAR_SIFT_GLOH_ANG_GRID = 8;		//GLOH网格沿角度方向等分区间个数
 
const int SAR_SIFT_DES_ANG_BINS = 8;		//像素梯度方向在0-360度内等分区间个数
 
const float SAR_SIFT_RADIUS_DES = 12.0f;	//描述子邻域半径		
 
const int Mmax = 8;							//像素梯度方向在0-360度内等分区间个数
 
const double T = 100.0;							//sobel算子去除冗余特征点的阈值
 
const float SAR_SIFT_GLOH_RATIO_R1_R2 = 0.73f;//GLOH网格中间圆半径和外圆半径之比
 
const float SAR_SIFT_GLOH_RATIO_R1_R3 = 0.25f;//GLOH网格最内层圆半径和外圆半径之比
 
#define Feature_Point_Minimum 1500			  //输入图像特征点最小个数
 
#define We 0.2
 
#define Wn 0.5
 
#define Row_num 3
 
#define Col_num 3
 
#define SIFT_FIXPT_SCALE 48					//不理解，后面可查原论文
 
/************************sift类*******************************/
class mySift
{
 
public:
	//默认构造函数
	mySift(int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.03,
		double edgeThreshold = 10, double sigma = 1.6, bool double_size = true) :nfeatures(nfeatures),
		nOctaveLayers(nOctaveLayers), contrastThreshold(contrastThreshold),
		edgeThreshold(edgeThreshold), sigma(sigma), double_size(double_size) {}
 
 
	//获得尺度空间每组中间层数
	int get_nOctave_layers() { return nOctaveLayers; }
 
	//获得图像尺度是否扩大一倍
	bool get_double_size() { return double_size; }
 
	//计算金字塔组数
	int num_octaves(const Mat& image);
 
	//生成高斯金字塔第一组，第一层图像
	void create_initial_image(const Mat& image, Mat& init_image);
 
	//使用 sobel 算子创建高斯金字塔第一层图像，以减少冗余特征点
	void sobel_create_initial_image(const Mat& image, Mat& init_image);
 
	//创建高斯金字塔
	void build_gaussian_pyramid(const Mat& init_image, vector<vector<Mat>>& gauss_pyramid, int nOctaves);
 
	//创建高斯差分金字塔
	void build_dog_pyramid(vector<vector<Mat>>& dog_pyramid, const vector<vector<Mat>>& gauss_pyramid);
 
	//该函数生成高斯差分金字塔
	void amplit_orient(const Mat& image, vector<Mat>& amplit, vector<Mat>& orient, float scale, int nums);
 
	//DOG金字塔特征点检测
	void find_scale_space_extrema(const vector<vector<Mat>>& dog_pyr, const vector<vector<Mat>>& gauss_pyr,
		vector<KeyPoint>& keypoints);
 
	//DOG金字塔特征点检测，特征点方向细化版
	void find_scale_space_extrema1(const vector<vector<Mat>>& dog_pyr, vector<vector<Mat>>& gauss_pyr,
		vector<KeyPoint>& keypoints);
 
	//计算特征点的描述子
	void calc_sift_descriptors(const vector<vector<Mat>>& gauss_pyr, vector<KeyPoint>& keypoints,
		Mat& descriptors, const vector<Mat>& amplit, const vector<Mat>& orient);
 
	//构建空间尺度—主要是为了获取 amplit 和 orient 在使用 GLOH 描述子的时候使用
	void build_sar_sift_space(const Mat& image, vector<Mat>& sar_harris_fun, vector<Mat>& amplit, vector<Mat>& orient);
 
	//GLOH 计算一个特征描述子
	void calc_gloh_descriptors(const vector<Mat>& amplit, const vector<Mat>& orient, const vector<KeyPoint>& keys, Mat& descriptors);
 
	//特征点检测
	void detect(const Mat& image, vector<vector<Mat>>& gauss_pyr, vector<vector<Mat>>& dog_pyr, vector<KeyPoint>& keypoints,
		vector<vector<vector<float>>>& all_cell_contrasts,
		vector<vector<float>>& average_contrast, vector<vector<int>>& n_cells, vector<int>& num_cell, vector<vector<int>>& available_n,
		vector<int>& available_num, vector<KeyPoint>& final_keypoints,
		vector<KeyPoint>& Final_keypoints, vector<KeyPoint>& Final_Keypoints);
 
	//特征点描述
	void comput_des(const vector<vector<Mat>>& gauss_pyr, vector<KeyPoint>& final_keypoints, const vector<Mat>& amplit,
		const vector<Mat>& orient, Mat& descriptors);
 
 
private:
	int nfeatures;				//设定检测的特征点的个数值,如果此值设置为0，则不影响结果
	int nOctaveLayers;			//每组金字塔中间层数
	double contrastThreshold;	//对比度阈值（D(x)）
	double edgeThreshold;		//特征点边缘曲率阈值
	double sigma;				//高斯尺度空间初始层的尺度
	bool double_size;			//是否上采样原始图像
 
 
};//注意类结束的分号