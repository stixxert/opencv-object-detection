CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

set(CMAKE_CXX_STANDARD 20)

project( ObjectDetection )
find_package(OpenCV REQUIRED)

set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( ObjectDetection src/object_detection.cpp include/timer.h src/timer.cpp
		include/common.hpp)

add_executable( ImageManipulation src/image_manipulation.cpp include/common.hpp)

target_link_libraries( ObjectDetection ${OpenCV_LIBS})

target_link_libraries( ImageManipulation ${OpenCV_LIBS} )