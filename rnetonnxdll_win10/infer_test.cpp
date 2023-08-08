#include "inference_pipeline.h"
#include "model_factory.h"

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


    YoloModelVersion model_version_v8 = YoloModelVersion::V8;
    InferenceInterface* yolov8_model = ModelFactory::createModel(model_version_v8, model_path_yolov8);
    InferencePipeline pipeline_v8(yolov8_model);
    pipeline_v8.runInference(image_v8);

    // Clean up memory
    //delete yolov5_model;
    delete yolov8_model;

    //Pot* potDet = new Pot;
    //BlackHead* bhDet = new BlackHead();
    //string srcImagePath = "RedMap.jpg";
    //Mat src = imread(srcImagePath);
    //Mat result = potDet->PotDetect(src);

    return 0;
}