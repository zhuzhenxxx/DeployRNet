#pragma once
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;
extern "C"
{
	class __declspec(dllexport) FaceProcess {
	public:
		// �ж���������
		bool hasFace(cv::Mat image);
		// �����ָ�
		Mat faceSlice(Mat image);
		// ����
		void makePoint(const std::string& image_path, const std::string& output_image_path);
		// �������
		void dectZW(const std::string& image_path, const std::string& output_image_path);
	};

	/*���Ƽ��*/
	class __declspec(dllexport) Wrinkles
	{
	public:
		Mat getGreenChannel(Mat src);
		Mat sharpen(Mat src);
		Mat gamma(Mat src, double gamma);
		void wrinkles(Mat src, Mat& dst);
	private:

	};

	/*�߿���*/
	class __declspec(dllexport) Pot :public Wrinkles
	{
	public:
		Mat PotDetect(Mat src);
		int areaThreshold = 50; //��������ֵ ֵԽ�󣬹��˵���С�����Խ��
	private:
	};

	/*��ͷ���*/
	class __declspec(dllexport) BlackHead :public Wrinkles
	{
	public:
		Mat getRedChannel(Mat src);
		Mat BlackHeadDetect(Mat src);
	private:
	};

}