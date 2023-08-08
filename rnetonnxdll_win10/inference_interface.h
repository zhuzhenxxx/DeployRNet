#pragma once
#include <string>
#include <vector>
using namespace std;
extern "C"
{
    class __declspec(dllexport) InferenceInterface {
        public:
            virtual ~InferenceInterface() {}
            virtual void preprocess() = 0;
            virtual std::vector<float> infer(const std::string& input_path) = 0;
            virtual void postprocess(const std::vector<float>& output_data) = 0;
    };


}