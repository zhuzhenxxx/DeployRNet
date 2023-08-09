#include "utils.h"

bool FaceProcess::hasFace(cv::Mat image)
{
    cv::CascadeClassifier face_cascade;
    face_cascade.load("haarcascade_frontalface_default.xml"); // Use the correct path to your xml

    // Check if the cascade file was loaded
    if (face_cascade.empty()) {
        std::cerr << "Failed to load face detector!" << std::endl;
        return false;
    }

    std::vector<cv::Rect> faces;
    cv::Mat gray;

    // Convert the image to gray scale
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Detect faces
    face_cascade.detectMultiScale(gray, faces, 1.1, 4);

    // If faces are detected, return true
    if (faces.size() > 0) {
        return true;
    }
    else {
        return false;
    }
}

Mat FaceProcess::faceSlice(Mat image)
{
    //Mat image = imread(image_path, IMREAD_COLOR);

    if (image.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
    }

    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    Mat blurred_image;
    blur(gray_image, blurred_image, Size(3, 3));

    Mat edges;
    Canny(blurred_image, edges, 30, 90); // Lower the threshold for more edges

    dilate(edges, edges, Mat(), Point(-1, -1));

    Mat colored_edges = Mat::zeros(image.size(), CV_8UC4);
    for (int y = 0; y < colored_edges.rows; ++y) {
        for (int x = 0; x < colored_edges.cols; ++x) {
            if (edges.at<uchar>(y, x) > 0) {
                colored_edges.at<Vec4b>(y, x) = Vec4b(0, 255, 0, 255); // Green color, full alpha
            }
            else {
                colored_edges.at<Vec4b>(y, x) = Vec4b(0, 0, 0, 0); // Transparent
            }
        }
    }

    namedWindow("Edges", WINDOW_NORMAL);
    imshow("Edges", colored_edges);
    //imwrite("colored_edges.png", colored_edges);
    
    //imwrite(output_image_path, colored_edges);
    return colored_edges;
    //waitKey(0);
}

void FaceProcess::makePoint(const std::string& image_path, const std::string& output_image_path)
{
    // ��ȡͼ��
    cv::Mat image = cv::imread(image_path);

    // ת����HSV��ɫ�ռ�
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // �����ɫ����ɫ��Χ
    cv::Scalar lower_red(0, 100, 100);
    cv::Scalar upper_red(10, 255, 255);

    // ����������ͻ����ɫ����
    cv::Mat mask_red;
    cv::inRange(hsv, lower_red, upper_red, mask_red);

    // ����һ��͸����ͼ��4ͨ��������Alphaͨ����
    cv::Mat transparent_image(mask_red.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));

    // ���ҵ��ĺ�ɫ�����ϻ��ϻ�ɫ��Բ�㣬��ȷ��ÿ����֮����һ������
    int step = 3; // ���ò��������Ƶ�֮��ľ���
    for (int y = 0; y < mask_red.rows; y += step) {
        for (int x = 0; x < mask_red.cols; x += step) {
            if (mask_red.at<uchar>(y, x) > 0) {
                cv::circle(transparent_image, cv::Point(x, y), 1, cv::Scalar(0, 255, 255, 255), -1); // ��ɫ���뾶Ϊ2
            }
        }
    }

    // ����͸��ͼ��ΪPNG��ʽ
    cv::imwrite(output_image_path, transparent_image);
}

void FaceProcess::dectZW(const std::string& image_path, const std::string& output_image_path)
{
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    // Ԥ����ƽ��
    cv::GaussianBlur(img, img, cv::Size(9, 9), 1);

    // �ֲ�ֱ��ͼ���⻯
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2);
    clahe->apply(img, img);

    // ��Ե���
    cv::Mat edges;
    cv::Laplacian(img, edges, CV_8U, 3);

    // ��ֵ��
    cv::threshold(edges, edges, 10, 255, cv::THRESH_BINARY);

    // �ҵ�����
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // ����͸��ͼ��4ͨ��������Alphaͨ����
    cv::Mat transparent_image(img.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));

    // ���峤����ֵ
    double lengthThreshold = 50;

    for (const auto& contour : contours) {
        double length = cv::arcLength(contour, false); // ���������ĳ���
        if (length > lengthThreshold) {
            double area = cv::contourArea(contour); // �������
            cv::Scalar color;

            // �������ѡ����ɫ
            if (area > 100) {
                color = cv::Scalar(0, 0, 255, 255); // ��ɫ
            }
            else if (area > 50) {
                color = cv::Scalar(0, 255, 255, 255); // ��ɫ
            }
            else {
                color = cv::Scalar(0, 255, 0, 255); // ��ɫ
            }

            // �����������߿�����Ϊ1
            cv::drawContours(transparent_image, std::vector<std::vector<cv::Point>>(1, contour), -1, color, 1);
        }
    }

    // ����͸��ͼ��ΪPNG��ʽ
    cv::imwrite(output_image_path, transparent_image);
}

void Wrinkles::wrinkles(Mat src, Mat& dst)
{
    if (src.empty())
        std::cerr << "src image is empty" << std::endl;
    Mat srcHsvImage, mask;
    if (src.channels() == 3)
    {
        cvtColor(src, srcHsvImage, COLOR_RGB2HSV);
    }
    else
    {
        std::cerr << "src image is not 3 channels" << std::endl;
    }
    Mat blur;
    cv::medianBlur(src, blur, 3);
    Mat srcCopy = blur.clone();
    Mat roi = blur.clone();
    cvtColor(srcCopy, srcCopy, COLOR_BGR2GRAY);
    cvtColor(roi, roi, COLOR_BGR2GRAY);
    //����hsv��ɫ�ռ�������������
    cv::Scalar scalarL = cv::Scalar(0, 0, 109);
    cv::Scalar scalarH = cv::Scalar(179, 155, 255);
    cv::inRange(srcHsvImage, scalarL, scalarH, mask);
    cv::bitwise_and(srcCopy, mask, roi);
    //��ֵ��
    threshold(roi, roi, 128, 255, THRESH_BINARY);
    //sobel��Ե���
    Mat gradX, gradY;
    Scharr(roi, gradX, CV_16S, 1, 0);
    Scharr(roi, gradY, CV_16S, 0, 1);
    convertScaleAbs(gradX, gradX);  // calculates absolute values, and converts the result to 8-bit.
    convertScaleAbs(gradY, gradY);
    imshow("mask", mask);
}


Mat Wrinkles::getGreenChannel(Mat src)
{
    vector<Mat> channels;
    assert(src.data != NULL);
    split(src, channels);
    Mat imageGreen = channels.at(1);
    return imageGreen;
}

Mat Wrinkles::sharpen(Mat src)
{
    Mat blur_img, usm;
    GaussianBlur(src, blur_img, Size(0, 0), 25);
    addWeighted(src, 1.5, blur_img, -0.5, 0, usm);
    return usm;
}

Mat Wrinkles::gamma(Mat src, double gamma)
{
    Mat fI;
    src.convertTo(fI, CV_64F, 1 / 255.0, 0);
    //٤��任
    Mat dst;
    pow(fI, gamma, dst);
    return dst;
}

Mat Pot::PotDetect(Mat src)
{
    Mat result(src.rows, src.cols, CV_8UC4, cv::Scalar(255, 255, 255, 0)); //��ͨ��ͼ�񱣴�������ʼ͸����Ϊ0
    Mat green = Pot::getGreenChannel(src);
    Mat sharpen = Pot::sharpen(green);
    Mat binary;
    threshold(sharpen, binary, 20, 255, THRESH_BINARY_INV); //��ֵ��

    /*��̬ѧ������*/
    Mat element, opening;
    element = getStructuringElement(MORPH_RECT, Size(7, 7));
    morphologyEx(binary, opening, MORPH_CLOSE, element);

    /*��������*/
    vector<vector<cv::Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(opening, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    for (int t = 0; t < contours.size(); t++)
    {
        double area = contourArea(contours[t]);
        if (area < Pot::areaThreshold)
        {
            continue;
        }
        drawContours(result, contours, t, Scalar(255, 255, 0, 255), 1); // ���Ƶ�������Ϊ��͸��
    }

    // ����ΪPNG����֧��͸����
    vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    cv::imwrite("result.png", result, compression_params); //������ͼ��

    return result;
}

Mat BlackHead::getRedChannel(Mat src)
{
    vector<Mat> channels;
    assert(src.data != NULL);
    split(src, channels);
    Mat imageRed = channels.at(0);
    return imageRed;
    return Mat();
}

Mat BlackHead::BlackHeadDetect(Mat src)
{
    Pot* pot = new Pot();
    Mat result(src.rows, src.cols, CV_8UC4, cv::Scalar(255, 255, 255, 255)); //��ͨ��ͼ�񱣴���
    return result;
}
