#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    const std::string pipeline = 
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
        "nvvidconv flip-method=2 ! "
        "video/x-raw,width=1280,height=720,format=BGRx ! "
        "videoconvert ! "
        "appsink";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cv::imshow("CSI Camera", frame);
        if (cv::waitKey(1) == 27) break; // ESC键退出
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}