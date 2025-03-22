#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <opencv2/opencv.hpp>

inline std::string toLower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
        [](unsigned char c) { return std::tolower(c); });

    return result;
}


inline cv::Mat getIdentityMatrix() {
    return cv::Mat::eye(3, 3, CV_64F);
}

inline cv::Mat getColumnVector(const double u, const double v) {
    return {3, 1, CV_64F, {u, v, 1}};
}

inline cv::Mat getColumnVector(const int u, const int v) {
    return getColumnVector(static_cast<double>(u), static_cast<double>(v));
}

template<typename T>
cv::Mat getColumnVector(cv::Point_<T> point) {
    return getColumnVector(point.x, point.y);
}

inline void convertToCoordinate(const cv::Mat& mat, int &u, int &v) {
    u = static_cast<int>(mat.at<double>(0) / mat.at<double>(2) + 0.5);
    v = static_cast<int>(mat.at<double>(1) / mat.at<double>(2) + 0.5);
}

inline cv::Point convertToCoordinate(const cv::Point2f& src) {
    int u, v;

    convertToCoordinate(getColumnVector(src.x, src.y), u, v);

    return {u, v};
}

inline cv::Mat translationMatrix(const double dx, const double dy) {
    cv::Mat mat = cv::Mat::eye(3, 3, CV_64F);

    mat.at<double>(0, 2) = dx;
    mat.at<double>(1, 2) = dy;

    return mat;
}

#endif //COMMON_H
