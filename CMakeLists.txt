cmake_minimum_required(VERSION 3.5)
project(CriminalID)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui objdetect)

# face module is in opencv_contrib - include it if available
find_package(OpenCV COMPONENTS face QUIET)
if(TARGET ${OpenCV_LIBS})
    message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(train_recognizer src/train_recognizer.cpp src/database.cpp)
target_link_libraries(train_recognizer ${OpenCV_LIBS} ${OpenCV_LIBS})

add_executable(recognize src/recognize.cpp src/database.cpp)
target_link_libraries(recognize ${OpenCV_LIBS} ${OpenCV_LIBS})

add_executable(detect_only src/detect_only.cpp)
target_link_libraries(detect_only ${OpenCV_LIBS})
