#include "yolo_models.h"
#include <fstream>
std::string labels_txt_file = "../classes.txt";

YoloV8::YoloV8(const std::string& model_path) : env(ORT_LOGGING_LEVEL_ERROR, "yolov8seg-onnx") 
{    
	onnxpath = model_path;
    std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);
    session = Ort::Session(env, modelPath.c_str(), session_options);

    classNames = readClassNames("../classes.txt");

    // Query input data format
    input_nodes_num = session.GetInputCount();
    output_nodes_num = session.GetOutputCount();
    auto allocator = Ort::AllocatorWithDefaultOptions();
    for (int i = 0; i < input_nodes_num; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        input_h = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()[2];
        input_w = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()[3];
        std::cout << "Input format: " << input_name << ", " << input_h << "x" << input_w << std::endl;
    }

    for (int i = 0; i < output_nodes_num; i++) {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(output_name.get());
        auto outShapeInfo = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }
}

YoloV8::~YoloV8() {
    session.release();
	env.release();
}

void YoloV8::runInference(const std::string& image_path) {
    // Load and preprocess the image
    cv::Mat frame = cv::imread(image_path);

    cv::Mat blob;
    preprocessImage(frame, x_factor, y_factor, blob);

    // Set input data and run inference
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), blob.total() * blob.channels(), input_shape_info.data(), input_shape_info.size());
    Ort::AllocatorWithDefaultOptions allocator;

    for (int i = 0; i < input_nodes_num; i++) {
		auto input_name = session.GetInputNameAllocated(i, allocator);
		input_node_names.push_back(input_name.get());
        auto inputShapeInfo = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        int ch = inputShapeInfo[1];
        input_h = inputShapeInfo[2];
        input_w = inputShapeInfo[3];
        std::cout << "input format: " << ch << "x" << input_h << "x" << input_w << std::endl;
    }

	for (int i = 0; i < output_nodes_num; i++) {
		auto output_name = session.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(output_name.get());
		auto outShapeInfo = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		if (i == 0)
		{
			num = outShapeInfo[1];
			nc = outShapeInfo[2];
		}
		else
		{
			output_h = outShapeInfo[2];
			output_w = outShapeInfo[3];
		}

	}

   /* const char* inputNames[] = { session.GetInputNameAllocated(0, allocator_info) };
    const char* outputNames[] = { session.GetOutputNameAllocated(0, allocator_info), session.GetOutputNameAllocated(1, allocator_info) };*/
    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 2> outNames = { output_node_names[0].c_str(), output_node_names[1].c_str() };

    try {
        ort_outputs = session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), & input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return;
    }

    // Post-process the results and display the image
    postprocessResults(frame, ort_outputs);
}

std::vector<std::string> YoloV8::readClassNames(const std::string& labels_txt_file) {
    std::vector<std::string> classNames;
    std::ifstream fp(labels_txt_file);
    if (!fp.is_open()) {
        printf("Could not open file...\n");
        exit(-1);
    }
    std::string name;
    while (!fp.eof()) {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back(name);
    }
    fp.close();
    return classNames;
}

float YoloV8::sigmoid_function(float a) {
    return 1.0f / (1.0f + exp(-a));
}

void YoloV8::preprocessImage(cv::Mat& frame, float& x_factor, float& y_factor, cv::Mat& blob) {
	start = cv::getTickCount();
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));
    x_factor = image.cols / static_cast<float>(input_h);
    y_factor = image.rows / static_cast<float>(input_w);
    blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
}

void YoloV8::postprocessResults(cv::Mat& frame, std::vector<Ort::Value>& ort_outputs) {
    // Display the results
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	const float* mdata = ort_outputs[1].GetTensorMutableData<float>();

	// 后处理, 1x116x8400, 84 - box, 80- min/max, 32 feature
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Mat> masks;
	cv::Mat dout(num, nc, CV_32F, (float*)pdata);
	cv::Mat det_output = dout.t(); // 116x8400 => 8400x116
	cv::Mat mask1(32, output_h * output_w, CV_32F, (float*)mdata);

	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, num - 32);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

		// 置信度 0～1之间
		if (score > 0.25)
		{
			cv::Mat mask2 = det_output.row(i).colRange(num - 32, num);
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;

			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
			masks.push_back(mask2);
		}
	}

	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	cv::Mat rgb_mask = cv::Mat::zeros(frame.size(), frame.type());

	sx = (float)output_h / input_h;
	sy = (float)output_w / input_w;

	// 显示处理
	for (size_t i = 0; i < indexes.size(); i++) {
		int idx = indexes[i];
		int cid = classIds[idx];
		cv::Rect box = boxes[idx];
		int x1 = std::max(0, box.x);
		int y1 = std::max(0, box.y);
		int x2 = std::max(0, box.br().x);
		int y2 = std::max(0, box.br().y);
		cv::Mat m2 = masks[idx];
		cv::Mat m = m2 * mask1;
		for (int col = 0; col < m.cols; col++) {
			m.at<float>(0, col) = sigmoid_function(m.at<float>(0, col));
		}
		cv::Mat m1 = m.reshape(1, output_h);
		int mx1 = std::max(0, int((x1 * sx) / x_factor));
		int mx2 = std::max(0, int((x2 * sx) / x_factor));
		int my1 = std::max(0, int((y1 * sy) / y_factor));
		int my2 = std::max(0, int((y2 * sy) / y_factor));

		// fix out of range box boundary on 2022-12-14
		if (mx2 >= m1.cols) {
			mx2 = m1.cols - 1;
		}
		if (my2 >= m1.rows) {
			my2 = m1.rows - 1;
		}
		// end fix it!!

		cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
		cv::Mat rm, det_mask;
		cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));
		for (int r = 0; r < rm.rows; r++) {
			for (int c = 0; c < rm.cols; c++) {
				float pv = rm.at<float>(r, c);
				if (pv > 0.5) {
					rm.at<float>(r, c) = 1.0;
				}
				else {
					rm.at<float>(r, c) = 0.0;
				}
			}
		}
		rm = rm * rng.uniform(0, 255);
		rm.convertTo(det_mask, CV_8UC1);
		if ((y1 + det_mask.rows) >= frame.rows) {
			y2 = frame.rows - 1;
		}
		if ((x1 + det_mask.cols) >= frame.cols) {
			x2 = frame.cols - 1;
		}

		cv::Mat mask = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
		det_mask(cv::Range(0, y2 - y1), cv::Range(0, x2 - x1)).copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));
		add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask);
		//cv::rectangle(frame, boxes[idx], cv::Scalar(0, 0, 255), 2, 8, 0);
		std::vector<std::string> labels = readClassNames(labels_txt_file);
		putText(frame, labels[cid].c_str(), boxes[idx].tl(), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 1, 8);
	}
	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

	cv::Mat result;
	cv::addWeighted(frame, 0.5, rgb_mask, 0.5, 0, result);
	result.copyTo(frame);

	std::cout << "frame wight: " << frame.cols << " height: " << frame.rows << std::endl;

	cv::imshow("ONNXRUNTIME1.13 + YOLOv8实例分割推理演示", frame);
	cv::imwrite("E:/codes/DeployRNet/test_image/1_out.jpg", frame);
	cv::waitKey(0);
}