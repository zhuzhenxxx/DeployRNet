
#include <iostream>
//#include <windows.h>
//#include <chrono>
//#include <ctime>
//#include <string>
//#include <filesystem>
//#include <thread>
//#include <opencv2/opencv.hpp>
//#include "depolyonnx.h"
//#define uInt8 unsigned int
//using namespace std;
//
//std::string input_path1 = "E:/codes/DeployRNet/x64/Release/111.jpg";
//std::vector<std::string> models_name_img1 = {
//	"E:/codes/DeployRNet/x64/Release/test2.onnx", 
//	//"E:/codes/DeployRNet/x64/Release/test2.onnx"
//};
//
//std::string input_path2 = "E:/codes/DeployRNet/x64/Release/111.jpg";
//std::vector<std::string> models_name_img2 = {
//	"E:/codes/DeployRNet/x64/Release/test2.onnx",
//	//	"F:/codes/DeployRNet/x64/Release/test2.onnx"
//};
//
//std::string prefix;
//
//void infer_test(std::string s, std::string input_path)
//{
//	size_t img_lastDotPos = input_path1.find_last_of('.');
//	size_t img_lastSlashPos = input_path1.find_last_of("/\\");
//	std::string img_name_without_extension = input_path1.substr(img_lastSlashPos + 1, img_lastDotPos - img_lastSlashPos - 1);
//
//	size_t s_lastDotPos = s.find_last_of('.');
//	size_t s_lastSlashPos = s.find_last_of("/\\");
//
//	std::string model_name_without_extension = s.substr(s_lastSlashPos + 1, s_lastDotPos - s_lastSlashPos - 1);
//	std::cout << " model_name_without_extension " << model_name_without_extension << std::endl;
//	std::string out_full_path = prefix + model_name_without_extension + "_" + img_name_without_extension + ".jpg";
//	std::cout << "out_full_path " << out_full_path << std::endl;
//
//	OnnxInfer infer(s, 4);
//	infer.Init();
//	infer.PrintModelInfo();
//
//	auto start = std::chrono::high_resolution_clock::now();
//	infer.RunInfer(input_path1, out_full_path);
//	auto end = std::chrono::high_resolution_clock::now();
//
//	//// Calculate the elapsed time
//	auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
//
//	std::cout << "image: " << img_name_without_extension << "model:" << model_name_without_extension << "Time taken by RunInfer: " << duration << " seconds" << std::endl;
//}
//
//int main()
//{
//	auto currentTime = std::chrono::system_clock::now();
//	auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(currentTime.time_since_epoch()).count();
//	std::string folderName = std::to_string(timestamp);
//
//	std::filesystem::path folderPath = std::filesystem::current_path() / folderName;
//
//	if (std::filesystem::create_directory(folderPath))
//	{
//		std::cout << "mkdir sucess£∫" << folderPath << std::endl;
//	}
//	else
//	{
//		std::cerr << "mkdir failed£∫" << folderPath << std::endl;
//		return 1;
//	}
//	prefix = folderPath.string() + "\\";
//
//	std::vector<std::thread> vecOfThreads;
//
//	for (auto s : models_name_img1)
//	{
//		vecOfThreads.push_back(std::thread(infer_test, s, input_path1));
//	}
//
//	for (auto s : models_name_img2)
//	{
//		vecOfThreads.push_back(std::thread(infer_test, s, input_path2));
//	}
//
//	for (std::thread& th : vecOfThreads)
//	{
//		// If thread Object is Joinable then Join that thread.
//		if (th.joinable())
//			th.join();
//	}
//
//	return 0;
//}

#include "inference_pipeline.h"
#include "model_factory.h"
#include "utils.h"
#include <opencv2/opencv.hpp>

int main()
{
    std::string image_v5 = "E:/codes/DeployRNet/x64/Release/111.jpg";
    std::string image_v8 = "E:/codes/DeployRNet/test_image/1.jpg";
    std::string model_path_yolov8 = "E:/codes/DeployRNet/models/FaceCrop.onnx";
    std::string model_path_yolov5 = "E:/codes/DeployRNet/x64/Release/test2.onnx";

    //// Replace with your desired model version
    //YoloModelVersion model_version_v5 = YoloModelVersion::V5; 
    //// Create the inference model using the factory
    //InferenceInterface* yolov5_model = ModelFactory::createModel(model_version_v5, model_path_yolov5);
    //// Create the inference pipeline and set the model
    //InferencePipeline pipeline_v5(yolov5_model);
    //// Run the inference process
    //pipeline_v5.runInference(image_v5);

    // 1. Õ∆¿Ì
    YoloModelVersion model_version_v8 = YoloModelVersion::V8;
    InferenceInterface* yolov8_model = ModelFactory::createModel(model_version_v8, model_path_yolov8);
    InferencePipeline pipeline_v8(yolov8_model);
    cv::Mat v8_orin = pipeline_v8.runInference(image_v8);

    // 2. »À¡≥«–∏Ó
    FaceProcess fp;
    cv::Mat slice_result = fp.faceSlice(v8_orin);
    namedWindow("slice_face", WINDOW_NORMAL);
    cv::imshow("slice_face", slice_result);
    waitKey(0);

    // 3. xxx
    // 4. xxx
    // 5. xxx
    // 
 
    // Clean up memory
    //delete yolov5_model;
    delete yolov8_model;

    return 0;
}