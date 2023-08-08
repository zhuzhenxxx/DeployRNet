#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
extern "C"
{
    class __declspec(dllexport) InferenceInterface {
        public:
            virtual ~InferenceInterface() {}
            virtual void preprocess() = 0;
            virtual cv::Mat infer(const std::string& input_path) = 0;
            virtual cv::Mat postprocess(const std::vector<float>& output_data) = 0;
    };


}