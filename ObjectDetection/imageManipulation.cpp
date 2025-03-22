#include <opencv2/opencv.hpp>

#include "common.hpp"
#include <cmath>

void printUsage() {
    std::cout << "Usage: " << std::endl;
    std::cout << " ImageManipulation <target image> <output image> <rotate|illuminate|scale> <value>" << std::endl;
    std::cout << " <target image> an image of the object to be detected" << std::endl;
    std::cout << " <output image> an image of a scene to search for the object" << std::endl;
    std::cout << " <rotate> rotate an image [-360, 360]" << std::endl;
    std::cout << " <illuminate> darken or brighten an image [-255, 255]" << std::endl;
    std::cout << " <scale> scale up/down an image" << std::endl;
}

cv::Size getNewBounds(cv::Mat& mat, cv::Size& src) {
    std::array<cv::Point, 4> corners = {
        {
            {0, 0},
            {src.width, 0},
            {src.width, src.height},
            {0, src.height}
        }
    };

    std::array<cv::Point, 4> newCorners;

    cv::transform(corners, newCorners, mat);

    auto newBounds = cv::boundingRect(newCorners);

    if (src.height == newBounds.height && src.width == newBounds.width)
        return src;

    newBounds.height += 10;
    newBounds.width += 10;

    mat.at<double>(0, 2) += static_cast<double>(newBounds.width) / 2 - static_cast<double>(src.width) / 2;
    mat.at<double>(1, 2) += static_cast<double>(newBounds.height) / 2 - static_cast<double>(src.height) / 2;

    return {newBounds.width, newBounds.height};
}

cv::Mat rotate(cv::Mat& src, int value) {
    auto rotationMatrix = cv::getRotationMatrix2D({static_cast<float>(src.cols / 2.0), static_cast<float>(src.rows / 2.0)}, value, 1);

    auto oldSize = src.size();

    auto newSize = getNewBounds(rotationMatrix, oldSize);

    cv::Mat dst;

    cv::warpAffine(src, dst, rotationMatrix, newSize);

    return dst;
}

cv::Mat illuminate(cv::Mat& src, int value) {
    cv::Mat dst = src.clone();

    for (int v = 0; v < src.rows; ++v) {
        for (int u = 0; u < src.cols; ++u) {
            for (int c = 0; c < src.channels(); ++c)
                dst.at<cv::Vec3b>(v, u)[c] = cv::saturate_cast<unsigned char>(
                    src.at<cv::Vec3b>(v, u)[c] + value);
        }
    }

    return dst;
}

cv::Mat scale(cv::Mat& src, double value) {
    auto rotationMatrix = cv::getRotationMatrix2D({static_cast<float>(src.cols / 2.0), static_cast<float>(src.rows / 2.0)}, 0, value);

    auto oldSize = src.size();

    auto newSize = getNewBounds(rotationMatrix, oldSize);

    cv::Mat dst;

    cv::warpAffine(src, dst, rotationMatrix, newSize);

    return dst;
}

int main(int argc, char* argv[]) {

    if (argc < 5) {
        printUsage();
    }

    cv::Mat img = cv::imread(argv[1]);
    cv::Mat res;

    if (img.empty()) {
        std::cerr << "Failed to read image from " << argv[1] << std::endl;
        exit(-2);
    }

    const float value = std::stof(argv[4]);

    if (toLower(argv[3]) == "rotate") {
        res = rotate(img, static_cast<int>(value));
    }

    else if (toLower(argv[3]) == "illuminate") {
        res = illuminate(img, static_cast<int>(value));
    }

    else if (toLower(argv[3]) == "scale") {
        res = scale(img, static_cast<double>(value));
    }

    else {
        std::cout<< "Invalid arguments" << std::endl;
        return 0;
    }

    cv::imwrite(argv[2], res);
    std::cout << "Wrote image output to \'" << argv[2] << "\'" << std::endl;
}