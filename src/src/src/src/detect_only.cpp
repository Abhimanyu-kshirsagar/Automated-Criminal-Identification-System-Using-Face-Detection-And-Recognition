// detect_only.cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main(){
    std::string cascade_path = "data/haarcascades/haarcascade_frontalface_default.xml";
    CascadeClassifier face_cascade;
    if (!face_cascade.load(cascade_path)) {
        std::cerr << "Cannot load cascade: " << cascade_path << std::endl;
        return -1;
    }
    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;
    Mat frame, gray;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        std::vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 4);
        for (auto &r: faces) rectangle(frame, r, Scalar(255,0,0), 2);
        imshow("Detect", frame);
        if (waitKey(30) == 'q') break;
    }
    return 0;
}
