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

#endif //COMMON_H
