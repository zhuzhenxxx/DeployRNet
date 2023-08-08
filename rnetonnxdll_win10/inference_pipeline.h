#pragma once
#include "inference_interface.h"
#include <opencv2/opencv.hpp>

extern "C"
{
    class __declspec(dllexport) InferencePipeline {
    public:
        InferencePipeline(InferenceInterface* inference_model) : model(inference_model) {}

        void setModel(InferenceInterface* inference_model, std::string model_path) {
            model = inference_model;
        }

        cv::Mat runInference(const std::string& input_path) {
            model->preprocess();
            return model->infer(input_path);
            //model->postprocess(output_data);
        }

    private:
        InferenceInterface* model;
    };
}