#ifndef FSRCNN_MODULE_H
#define FSRCNN_MODULE_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>

class SuperResolution {
public:
    SuperResolution();
    bool loadModel(const std::string& model_path);
    bool processImage(const cv::Mat& src,  cv::Mat& dist);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
};

#endif // FSRCNN_MODULE_H
