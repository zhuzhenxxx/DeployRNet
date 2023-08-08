#pragma once
#include "yolo_models.h"

enum class YoloModelVersion {
    V5,
    V8
};

class ModelFactory {
public:
    static InferenceInterface* createModel(YoloModelVersion version, const std::string& model_path) {
        switch (version) {
        case YoloModelVersion::V5:
            return new YoloV5(model_path);
        case YoloModelVersion::V8:
            return new YoloV8(model_path);
        default:
            return nullptr; // Or throw an exception for unknown versions
        }
    }
};