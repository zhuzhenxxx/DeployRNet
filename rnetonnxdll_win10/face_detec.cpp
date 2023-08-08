//#include "inference_interface.h"
//
//void Wrinkles::wrinkles(Mat src, Mat& dst)
//{
//	if (src.empty())
//		std::cerr << "src image is empty" << std::endl;
//	Mat srcHsvImage, mask;
//	if (src.channels() == 3)
//	{
//		cvtColor(src, srcHsvImage, COLOR_RGB2HSV);
//	}
//	else
//	{
//		std::cerr << "src image is not 3 channels" << std::endl;
//	}
//	Mat blur;
//	cv::medianBlur(src, blur, 3);
//	Mat srcCopy = blur.clone();
//	Mat roi = blur.clone();
//	cvtColor(srcCopy, srcCopy, COLOR_BGR2GRAY);
//	cvtColor(roi, roi, COLOR_BGR2GRAY);
//	//根据hsv颜色空间区分脸部区域
//	cv::Scalar scalarL = cv::Scalar(0, 0, 109);
//	cv::Scalar scalarH = cv::Scalar(179, 155, 255);
//	cv::inRange(srcHsvImage, scalarL, scalarH, mask);
//	cv::bitwise_and(srcCopy, mask, roi);
//	//二值化
//	threshold(roi, roi, 128, 255, THRESH_BINARY);
//	//sobel边缘检测
//	Mat gradX, gradY;
//	Scharr(roi, gradX, CV_16S, 1, 0);
//	Scharr(roi, gradY, CV_16S, 0, 1);
//	convertScaleAbs(gradX, gradX);  // calculates absolute values, and converts the result to 8-bit.
//	convertScaleAbs(gradY, gradY);
//	imshow("mask", mask);
//}
//
//
//Mat Wrinkles::getGreenChannel(Mat src)
//{
//	vector<Mat> channels;
//	assert(src.data != NULL);
//	split(src, channels);
//	Mat imageGreen = channels.at(1);
//	return imageGreen;
//}
//
//Mat Wrinkles::sharpen(Mat src)
//{
//	Mat blur_img, usm;
//	GaussianBlur(src, blur_img, Size(0, 0), 25);
//	addWeighted(src, 1.5, blur_img, -0.5, 0, usm);
//	return usm;
//}
//
//Mat Wrinkles::gamma(Mat src, double gamma)
//{
//	Mat fI;
//	src.convertTo(fI, CV_64F, 1 / 255.0, 0);
//	//伽马变换
//	Mat dst;
//	pow(fI, gamma, dst);
//	return dst;
//}
//
//Mat Pot::PotDetect(Mat src)
//{
//	Mat result(src.rows, src.cols, CV_8UC4, cv::Scalar(255, 255, 255, 0)); //四通道图像保存结果，初始透明度为0
//	Mat green = Pot::getGreenChannel(src);
//	Mat sharpen = Pot::sharpen(green);
//	Mat binary;
//	threshold(sharpen, binary, 10, 255, THRESH_BINARY_INV); //二值化
//
//	/*形态学开运算*/
//	Mat element, opening;
//	element = getStructuringElement(MORPH_RECT, Size(3, 3));
//	morphologyEx(binary, opening, MORPH_CLOSE, element);
//
//	/*绘制轮廓*/
//	vector<vector<cv::Point>> contours;
//	vector<Vec4i> hierarchy;
//	findContours(opening, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
//	for (int t = 0; t < contours.size(); t++)
//	{
//		double area = contourArea(contours[t]);
//		if (area < Pot::areaThreshold)
//		{
//			continue;
//		}
//		drawContours(result, contours, t, Scalar(255, 255, 0, 255), 1); // 绘制的轮廓设为不透明
//	}
//
//	// 保存为PNG，以支持透明度
//	vector<int> compression_params;
//	compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
//	compression_params.push_back(9);
//	cv::imwrite("result.png", result, compression_params); //保存结果图像
//
//	return result;
//}
//
//Mat BlackHead::getRedChannel(Mat src)
//{
//	vector<Mat> channels;
//	assert(src.data != NULL);
//	split(src, channels);
//	Mat imageRed = channels.at(0);
//	return imageRed;
//	return Mat();
//}
//
//Mat BlackHead::BlackHeadDetect(Mat src)
//{
//	Pot* pot = new Pot();
//	Mat result(src.rows, src.cols, CV_8UC4, cv::Scalar(255, 255, 255, 255)); //四通道图像保存结果
//	return result;
//}
