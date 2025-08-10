// database.cpp
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>

std::unordered_map<int, std::string> load_labels(const std::string &csv_path) {
    std::unordered_map<int, std::string> labels;
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Could not open labels file: " << csv_path << "\n";
        return labels;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string id_s, name;
        if (!std::getline(ss, id_s, ',')) continue;
        if (!std::getline(ss, name)) continue;
        int id = std::stoi(id_s);
        labels[id] = name;
    }
    return labels;
}
