// train_recognizer.cpp
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include "database.cpp" // loads labels.csv

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::face;
using std::cout;
using std::endl;

int main(int argc, char** argv) {
    std::string data_dir = "data/criminals";
    std::string cascade_path = "data/haarcascades/haarcascade_frontalface_default.xml";
    std::string labels_csv = "data/labels.csv";
    std::string out_model = "models/lbph_model.yml";

    if (argc >= 2) data_dir = argv[1];
    if (argc >= 3) cascade_path = argv[2];

    CascadeClassifier face_cascade;
    if (!face_cascade.load(cascade_path)) {
        std::cerr << "Error loading cascade at " << cascade_path << std::endl;
        return -1;
    }

    std::vector<Mat> images;
    std::vector<int> labels;

    // Iterate over subfolders in data/criminals (each subfolder name is label_id)
    for (const auto &entry : fs::directory_iterator(data_dir)) {
        if (!entry.is_directory()) continue;
        std::string label_dir = entry.path().string();
        std::string label_name = entry.path().filename().string();
        int label_id = std::stoi(label_name); // expect directories named like 1,2,3...

        for (const auto &imgf : fs::directory_iterator(label_dir)) {
            std::string img_path = imgf.path().string();
            Mat img = imread(img_path);
            if (img.empty()) {
                std::cerr << "Could not read image: " << img_path << std::endl;
                continue;
            }
            Mat gray;
            cvtColor(img, gray, COLOR_BGR2GRAY);
            std::vector<Rect> faces;
            face_cascade.detectMultiScale(gray, faces, 1.1, 4, 0, Size(80,80));
            if (faces.empty()) {
                std::cerr << "No face detected in " << img_path << " - skipping\n";
                continue;
            }
            // take largest face
            Rect best = faces[0];
            for (auto &r : faces) if (r.area() > best.area()) best = r;
            Mat face = gray(best).clone();
            // optional: resize to fixed size
            resize(face, face, Size(200,200));
            images.push_back(face);
            labels.push_back(label_id);
            cout << "Added: " << img_path << " label=" << label_id << "\n";
        }
    }

    if (images.empty()) {
        std::cerr << "No training images found. Exiting.\n";
        return -1;
    }

    // Create LBPH recognizer (parameters can be tuned)
    Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create(1,8,8,8,100.0);
    recognizer->train(images, labels);

    // Create models dir if needed
    fs::create_directories("models");
    recognizer->save(out_model);
    cout << "Model saved to " << out_model << "\n";

    // Optional: save labels mapping (copy labels.csv to models or use database.csv)
    cout << "Training complete. Trained on " << images.size() << " faces.\n";
    return 0;
}
