#pragma once
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

class FaceProcess {
	// 判断有无人脸
	bool hasFace(cv::Mat image);
	// 人脸分割
	void faceSlice(const std::string& image_path, const std::string& output_image_path);
	// 画点
	void makePoint(const std::string& image_path, const std::string& output_image_path);
	// 检测皱纹
	void dectZW(const std::string& image_path, const std::string& output_image_path);
};

/*皱纹检测*/
class Wrinkles
{
public:
	Mat getGreenChannel(Mat src);
	Mat sharpen(Mat src);
	Mat gamma(Mat src, double gamma);
	void wrinkles(Mat src, Mat& dst);
private:

};

/*斑块检测*/
class Pot :public Wrinkles
{
public:
	Mat PotDetect(Mat src);
	int areaThreshold = 50; //红斑面积阈值 值越大，过滤掉的小面积斑越多
private:
};

/*黑头检测*/
class BlackHead :public Wrinkles
{
public:
	Mat getRedChannel(Mat src);
	Mat BlackHeadDetect(Mat src);
private:
};