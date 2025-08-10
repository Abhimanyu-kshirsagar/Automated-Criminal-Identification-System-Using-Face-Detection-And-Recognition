// recognize.cpp
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include "database.cpp"

using namespace cv;
using namespace cv::face;
using std::cout;
using std::endl;

int main(int argc, char** argv) {
    std::string cascade_path = "data/haarcascades/haarcascade_frontalface_default.xml";
    std::string model_path = "models/lbph_model.yml";
    std::string labels_csv = "data/labels.csv";

    if (argc >= 2) model_path = argv[1];

    CascadeClassifier face_cascade;
    if (!face_cascade.load(cascade_path)) {
        std::cerr << "Error loading cascade: " << cascade_path << std::endl;
        return -1;
    }

    Ptr<LBPHFaceRecognizer> recognizer;
    try {
        recognizer = LBPHFaceRecognizer::create();
        recognizer->read(model_path);
    } catch (const cv::Exception &e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    auto labels_map = load_labels(labels_csv);

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening camera\n";
        return -1;
    }

    Mat frame, gray;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        std::vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 4, 0, Size(80,80));

        for (const auto &r : faces) {
            Mat face = gray(r).clone();
            resize(face, face, Size(200,200));
            int predicted_label = -1;
            double confidence = 0;
            recognizer->predict(face, predicted_label, confidence);
            std::string name = "Unknown";
            if (labels_map.find(predicted_label) != labels_map.end()) {
                name = labels_map[predicted_label];
            }
            rectangle(frame, r, Scalar(0,255,0), 2);
            std::ostringstream text;
            text << name << " (" << (int)confidence << ")";
            putText(frame, text.str(), Point(r.x, r.y-10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 2);
        }

        imshow("Criminal ID - Press q to quit", frame);
        char c = (char)waitKey(30);
        if (c == 'q' || c == 27) break;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
