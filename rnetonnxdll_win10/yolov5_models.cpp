#include "yolo_models.h"
#include <iostream>
#include <assert.h>
#include <algorithm>
#include <time.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <string>
#include <locale>
#include <codecvt>
#include <cstring>
#include <winsock.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

#define N 999
#define MAXTHREADCOUNTS 8
#define MINTHREADCOUNTS 1
typedef unsigned char uInt8;
using namespace std::chrono;
using namespace std;

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
	os << "[";
	for (int i = 0; i < v.size(); ++i)
	{
		os << v[i];
		if (i != v.size() - 1)
		{
			os << ", ";
		}
	}
	os << "]";
	return os;
}

std::ostream& operator<<(std::ostream& os,
	const ONNXTensorElementDataType& type)
{
	switch (type)
	{
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
		os << "undefined";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		os << "float";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		os << "uint8_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		os << "int8_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
		os << "uint16_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		os << "int16_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		os << "int32_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		os << "int64_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
		os << "std::string";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
		os << "bool";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
		os << "float16";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		os << "double";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
		os << "uint32_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
		os << "uint64_t";
		break;
	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
		os << "float real + float imaginary";
		break;
		case ONNXTensorElementDataType::
		ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
			os << "double real + float imaginary";
			break;
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
			os << "bfloat16";
			break;
		default:
			break;
	}

	return os;
}


wchar_t* char2wchar(const char* cchar)
{
	wchar_t* m_wchar;
	int len = MultiByteToWideChar(CP_ACP, 0, cchar, strlen(cchar), NULL, 0);
	m_wchar = new wchar_t[len + 1];
	MultiByteToWideChar(CP_ACP, 0, cchar, strlen(cchar), m_wchar, len);
	m_wchar[len] = '\0';
	return m_wchar;
}

inline void appendlog(string& log, char* str) {
	log.append(str);
	memset(str, 0, sizeof(str));
}

/*
DET识别前处理: HWC -> CHW , Normalization, BGR -> RGB, 该模式只能在主程序进行
*/
int YoloV5::asyc_task_for_det_bgr(int c, uint8_t* ptMat)
{
	for (int h = 0; h < input_height_; ++h)
	{
		for (int w = 0; w < input_width_; ++w)
		{
			size_t dstIdx = c * input_height_ * input_width_ + h * input_width_ + w;
			size_t srcIdx = h * input_width_ * input_channel_ + w * input_channel_ + 2 - c;
			//fileData_prec[dstIdx] = static_cast<float>((ptMat[srcIdx] * 1.0f / 255.0f - MEAN_RGB[c]) * 1.0f / STD_RGB[c]);
			feed_img_data_[dstIdx] = static_cast<float>(((ptMat[srcIdx] * pxilefloat) - MEAN_RGB[2 - c]) / STD_RGB[2 - c]); //像素值归一化
		}
	}
	return 0;
}
/*
DET识别前处理: HWC -> CHW , Normalization, RGB -> RGB
*/
int YoloV5::asyc_task_for_det_rgb(int c, uint8_t* ptMat)
{
	for (int h = 0; h < input_height_; ++h)
	{
		for (int w = 0; w < input_width_; ++w)
		{
			size_t dstIdx = c * input_height_ * input_width_ + h * input_width_ + w;
			size_t srcIdx = h * input_width_ * input_channel_ + w * input_channel_ + c;
			//fileData_prec[dstIdx] = static_cast<float>((ptMat[srcIdx] * 1.0f / 255.0f - MEAN_RGB[c]) * 1.0f / STD_RGB[c]);
			feed_img_data_[dstIdx] = static_cast<float>(((ptMat[srcIdx] * pxilefloat) - MEAN_RGB[c]) / STD_RGB[c]); //像素值归一化
		}
	}
	return 0;
}


// RGB -> BGR & CHW -> HWC
int YoloV5::asyc_task_for_chw2hwc_abnormal(int c)
{
	//优化for循环顺序，先循环c
	for (int h = 0; h < output_height_; ++h)
	{
		for (int w = 0; w < output_width_; ++w)
		{
			int dstIdx = h * output_width_ * output_channel_ + w * output_channel_ + 2 - c;
			int srcIdx = c * output_height_ * output_width_ + h * output_width_ + w;
			float tmp_pix = (fill_output_[srcIdx] * STD_RGB[c] + MEAN_RGB[c]) * 255;
			fill_output_hwc_abnormal_[dstIdx] = static_cast<uint8_t>(tmp_pix > 255 ? 255 : tmp_pix);
		}
	}
	return 0;
}


int YoloV5::task_for_chw2hwc_abnormal()
{
	//优化for循环顺序，先循环c
	fill_output_hwc_abnormal_.resize(output_height_ * output_width_ * output_channel_);
	for (int h = 0; h < output_height_; ++h)
	{
		for (int w = 0; w < output_width_; ++w)
		{
			for (int c = 0; c < output_channel_; ++c)
			{
				int dstIdx = h * output_width_ * output_channel_ + w * output_channel_ + 2 - c;
				int srcIdx = c * output_height_ * output_width_ + h * output_width_ + w;
				float tmp_pix = (fill_output_[srcIdx] * STD_RGB[c] + MEAN_RGB[c]) * 255;
				fill_output_hwc_abnormal_[dstIdx] = static_cast<uint8_t>(tmp_pix > 255 ? 255 : tmp_pix);
			}
		}
	}
	return 0;
}


YoloV5::YoloV5(const std::string& model_path) : model_{ model_path }
{
	
}

int YoloV5::Init()
{
	try
	{
		const wchar_t* modelpath_wchar = char2wchar(model_.data());
		if (threadcounts_ == MINTHREADCOUNTS || threadcounts_ > MAXTHREADCOUNTS)
			threadcounts_ = MINTHREADCOUNTS;

		session_options = new Ort::SessionOptions();
		env = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov5_dect");

		session_options->SetIntraOpNumThreads(threadcounts_);
		session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		session_options->EnableCpuMemArena();
		session_options->EnableMemPattern();
		session_options->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL); // only use ORT_SEQUENTIAL

		//session = new Ort::Session(*env, modelpath_wchar, *session_options);

		session = Ort::Session(*env, modelpath_wchar, *session_options);
		//info << "onnx model loaded successfully...\n";

		Ort::AllocatorWithDefaultOptions allocator;
		Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
		input_name_ = input_name_ptr.get();
		Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
		output_name_ = output_name_ptr.get();

		Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
		auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
		std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

		info << std::setw(40) << "infer thread counts: " << std::setw(10) << threadcounts_ << std::endl;
		info << std::setw(40) << "input name: " << std::setw(10) << input_name_ << std::endl;
		info << std::setw(40) << "output name: " << std::setw(10) << output_name_ << std::endl;
		info << std::setw(40) << "input Dimensions:" << std::setw(5) << inputDims << std::endl;


		if (inputDims.size() == 4)
		{
			if (inputDims[1] > 1 && inputDims[2] > 1 && inputDims[3] > 1)
			{

				info << std::setw(40) << "Input dimension format: " << std::setw(10) << "NCHW" << std::endl;

				input_channel_ = inputDims[1];
				input_height_ = inputDims[2];
				input_width_ = inputDims[3];
				input_bs_ = inputDims[0];
				input_shape_ = { input_bs_, input_channel_, input_height_, input_width_ };
			}
			else
			{

				info << std::setw(40) << "Input dimension format: " << std::setw(10) << "NHWC" << std::endl;

				input_channel_ = inputDims[3];
				input_height_ = inputDims[1];
				input_width_ = inputDims[2];
				input_bs_ = inputDims[0];
				input_shape_ = { input_bs_, input_height_, input_width_, input_channel_ };
			}
		}
		else
		{
			std::cout << "Unsupported Input dimension format" << std::endl;
		}

		input_len_ = input_bs_ * input_channel_ * input_height_ * input_width_ * sizeof(uint16_t);
		output_len_ *= sizeof(uint16_t);
		Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
		auto OutputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType outputType = OutputTensorInfo.GetElementType();

		std::vector<int64_t> outputDims = OutputTensorInfo.GetShape();

		info << std::setw(40) << "output Dimensions: " << std::setw(5) << outputDims << std::endl;

		if (outputDims.size() == 4)
		{
			if (outputDims[1] > 1 && outputDims[2] > 1 && outputDims[3] > 1)
			{
				info << std::setw(40) << "Output dimension format: " << std::setw(10) << "NCHW" << std::endl;
				output_channel_ = outputDims[1];
				output_height_ = outputDims[2];
				output_width_ = outputDims[3];
				output_bs_ = outputDims[0];
				output_shape_ = { output_bs_, output_channel_, output_height_, output_width_ };
			}
			else
			{
				info << std::setw(40) << "Output dimension format: " << std::setw(10) << "NHWC" << std::endl;
				output_channel_ = outputDims[3];
				output_height_ = outputDims[1];
				output_width_ = outputDims[2];
				output_bs_ = outputDims[0];
				output_shape_ = { output_bs_, output_height_, output_width_, output_channel_ };
			}
		}
		else
		{
			std::cout << "Unsupported input dimension format" << std::endl;
		}

		output_len_ = output_bs_ * output_channel_ * output_height_ * output_width_;

		info << std::setw(40) << "Input Type: " << std::setw(10) << inputType << std::endl;
		info << std::setw(40) << "Output Type: " << std::setw(10) << outputType << std::endl;

	}
	catch (const std::exception& e)
	{
		std::cout << "init failed. " << e.what() << std::endl;
		return -1;
	}
	return 0;
}

void YoloV5::PrintModelInfo()
{
	std::cout << info.str() << std::endl;
}


void YoloV5::GetOutput_LVM(char* save_filepath_abs)
{
	std::future<int> futu0_det = std::async(&YoloV5::asyc_task_for_chw2hwc_abnormal, this, 0);
	std::future<int> futu1_det = std::async(&YoloV5::asyc_task_for_chw2hwc_abnormal, this, 1);
	std::future<int> futu2_det = std::async(&YoloV5::asyc_task_for_chw2hwc_abnormal, this, 2);

	futu0_det.get();
	futu1_det.get();
	futu2_det.get();

	if (fill_output_hwc_abnormal_.empty())
	{
		std::cout << "output data is empty. " << std::endl;
		return;
	}

	cv::Mat srcMat(output_height_, output_width_, CV_8UC3, fill_output_hwc_abnormal_.data());
	cv::imwrite(save_filepath_abs, srcMat);

}

cv::Mat YoloV5::RunInfer(const std::string& input_img_path, const std::string& output_img_path)
{
	if (input_img_path.empty())
	{
		std::cout << "input_img_path is null." << std::endl;
	}
	cv::Mat imageBGR = cv::imread(input_img_path);
	cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
	cv::resize(imageBGR, resizedImageBGR,
		cv::Size(input_width_, input_height_),
		cv::InterpolationFlags::INTER_CUBIC);
	cv::cvtColor(resizedImageBGR, resizedImageRGB,
		cv::ColorConversionCodes::COLOR_BGR2RGB);  //BGR2RGB

	cv::Mat img_prc = resizedImageRGB.clone();
	uint8_t* ptMat_det = img_prc.ptr<uint8_t>(0);
	feed_img_data_.resize(input_height_ * input_width_ * input_channel_);
	std::future<int> futu0_det = std::async(std::launch::async, &YoloV5::asyc_task_for_det_rgb, this, 0, ptMat_det);
	std::future<int> futu1_det = std::async(std::launch::async, &YoloV5::asyc_task_for_det_rgb, this, 1, ptMat_det);
	std::future<int> futu2_det = std::async(std::launch::async, &YoloV5::asyc_task_for_det_rgb, this, 2, ptMat_det);
	futu0_det.get();
	futu1_det.get();
	futu2_det.get();
	if (feed_img_data_.empty())
	{
		std::cout << "Invalid image format. Must be 224*224 RGB image. " << std::endl;
	}

	try
	{

		Ort::IoBinding io_binding(session);
		Ort::Value input_tensor_{ Ort::Value::CreateTensor<float>(input_mem_info_, feed_img_data_.data(), input_len_, input_shape_.data(), input_shape_.size()/*, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16*/)};
		fill_output_.resize(output_len_);
		Ort::Value output_tensor_{ Ort::Value::CreateTensor<float>(output_mem_info_, fill_output_.data(), output_len_, output_shape_.data(), output_shape_.size()/*, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16*/)};
		io_binding.BindInput(input_name_.c_str(), input_tensor_);
		io_binding.BindOutput(output_name_.c_str(), output_tensor_);
		session.Run(Ort::RunOptions{ nullptr }, io_binding);

		cv::Mat chwImage(output_height_, output_width_, CV_16UC3, fill_output_.data());
		fill_output_hwc_abnormal_.resize(output_len_);
		std::future<int> det_out0 = std::async(std::launch::async, &YoloV5::asyc_task_for_chw2hwc_abnormal, this, 0);
		std::future<int> det_out1 = std::async(std::launch::async, &YoloV5::asyc_task_for_chw2hwc_abnormal, this, 1);
		std::future<int> det_out2 = std::async(std::launch::async, &YoloV5::asyc_task_for_chw2hwc_abnormal, this, 2);
		det_out0.get();
		det_out1.get();
		det_out2.get();
		cv::Mat hwcImage(output_height_, output_width_, CV_8UC3, fill_output_hwc_abnormal_.data());
		cv::imwrite(output_img_path, hwcImage);
		return hwcImage;
	}
	catch (const Ort::Exception& exception)
	{
		cout << "error running model inference: " << exception.what() << endl;
	}
	cout << "success running model inference. " << "images: " << input_img_path << endl;
}

YoloV5::~YoloV5()
{
	if (env != nullptr)
	{
		env->release();
		delete env;
		env = nullptr;
	}
	if (session_options != nullptr)
	{
		session_options->release();
		delete session_options;
		session_options = nullptr;
	}
	//if (session != nullptr)
	//{
	//	session->release();
	//	delete session;
	//	session = nullptr;
	//}

	input_mem_info_.release();
	output_mem_info_.release();

	feed_img_data_.clear();
	vector<float> Elements;
	// fill the vector up
	vector<float>().swap(feed_img_data_);
}