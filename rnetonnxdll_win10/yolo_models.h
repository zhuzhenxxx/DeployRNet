#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"
#include "inference_interface.h"
#include <string>
#include <sstream>
//===============================================
#define _WIN32_WINNT_WIN7 0x0601 
//===========================================================================
#include <future>
#include <onnxruntime_cxx_api.h>
#define NONESET -999

//normalization info
constexpr float MEAN_RGB[3] = { 0.5f, 0.5f, 0.5f };
constexpr float STD_RGB[3] = { 0.5f, 0.5f, 0.5f };
constexpr float pxilefloat = 1.0f / 255.0f;
extern "C"
{
    class __declspec(dllexport) YoloV5 : public InferenceInterface {
        public:
            YoloV5(const std::string& model_path);
            ~YoloV5();

            void preprocess() override {
                // Implement YOLOv5 preprocessing logic here
                //model_path_ = "E:/codes/DeployRNet/x64/Release/test2.onnx";
                Init();
                PrintModelInfo();
            }

            cv::Mat infer(const std::string& input_path) override {
                // Implement YOLOv5 inference logic here
                std::string output = "E:/codes/DeployRNet/x64/Release/yolov5_out.jpg";
                return RunInfer(input_path, output);
            }

            cv::Mat postprocess(const std::vector<float>& output_data) override {
                // Implement YOLOv5 postprocessing logic here
                cv::Mat tmp;
                return tmp;
            }

        private:

            int asyc_task_for_det_bgr(int c, uint8_t* ptMat);
            int asyc_task_for_det_rgb(int c, uint8_t* ptMat);
            int asyc_task_for_chw2hwc_abnormal(int c);
            int task_for_chw2hwc_abnormal();
            void GetOutput_LVM(char* save_filepath_abs);

            int Init();
            void PrintModelInfo();
            cv::Mat RunInfer(const std::string& input_img_path, const std::string& output_img_path);
        private:

            std::string                 model_;
            int							threadcounts_{ 4 };
            Ort::Env* env{ nullptr };
            Ort::SessionOptions* session_options{ nullptr };
            Ort::Session				session{ nullptr };

            const size_t				num_input_nodes = 1;
            const size_t				num_output_nodes = 1;
            std::vector<const char*>	input_node_names;
            std::vector<int64_t>		input_node_dims;
            std::vector<const char*>	output_node_names;
            std::vector<int64_t>		output_node_dims;

            //input info
            std::string input_name_;	// 输入名
            std::string output_name_;	// 输出名

            int64_t input_channel_{ 0 };
            int64_t input_height_{ 0 };
            int64_t input_width_{ 0 };
            int64_t input_bs_{ 1 };
            std::array<int64_t, 4> input_shape_;
            int64_t input_len_{ 1 };
            std::vector<float> feed_img_data_;

            Ort::MemoryInfo input_mem_info_{ Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault) };
            Ort::MemoryInfo output_mem_info_{ Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault) };


            ONNXTensorElementDataType inputType;
            ONNXTensorElementDataType outputType;
            //output info, do not change
            int64_t output_channel_{ 0 };
            int64_t output_height_{ 0 };
            int64_t output_width_{ 0 };
            int64_t output_bs_{ 1 };
            std::array<int64_t, 4> output_shape_;
            int64_t output_len_{ 1 };
            std::vector<float> fill_output_;
            std::vector<uint8_t> fill_output_hwc_abnormal_;


            std::stringstream info;
    };

    class __declspec(dllexport) YoloV8 : public InferenceInterface {
    public:
        YoloV8(const std::string& model_path);
        ~YoloV8();

        cv::Mat runInference(const std::string& image_path);
        std::vector<std::string> readClassNames(const std::string& labels_txt_file);
        float sigmoid_function(float a);
        void preprocessImage(cv::Mat& frame, float& x_factor, float& y_factor, cv::Mat& blob);
        cv::Mat postprocessResults(cv::Mat& frame, std::vector<Ort::Value>& ort_outputs);
    public:
        void preprocess() override {
            // Implement YOLOv8 preprocessing logic here

        }

        cv::Mat infer(const std::string& input_path) override {
            // Implement YOLOv8 inference logic here
            return runInference(input_path);
            //std::vector<float> tmp;
            //return tmp;
        }

        cv::Mat postprocess(const std::vector<float>& output_data) override {
            // Implement YOLOv8 postprocessing logic here
            cv::Mat tmp;
            return tmp;
        }

    private:

        std::string onnxpath;
        std::string                 model_;
        Ort::Env env;
        Ort::Session session{nullptr};
        std::vector<std::string> classNames;
        int input_h;
        int input_w;

        int num = 0;
        int nc = 0;

        int output_h;
        int output_w;

        int input_nodes_num;
        int output_nodes_num;
        std::vector<std::string> input_node_names;
        std::vector<std::string> output_node_names;
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<Ort::Value> ort_outputs;

        float x_factor = 0.0;
        float y_factor = 0.0;

        float sx = 160.0f / input_h;
        float sy = 160.0f / input_w;

        int64 start;

        //cv::Mat rm, det_mask;
        cv::RNG rng;

    };
}