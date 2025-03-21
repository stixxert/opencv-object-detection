#include "Timer.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <string>

constexpr bool kDTreeEnabled = true;
constexpr int orbFeatures = 4000;

constexpr bool verbose = false;

template<typename T, size_t N>
using PointContainer = std::array<cv::Point_<T>, N>;

using CornerPointsContainerFloat = PointContainer<float, 4>;

using TransformPoints = std::vector<cv::Point2f>;
using TransformMatches = std::pair<TransformPoints, TransformPoints>;
using MatchesContainer = std::vector<cv::DMatch>;
using MatchesContainerKNN = std::vector<MatchesContainer>;
using FeaturePoints = std::vector<cv::KeyPoint>;

struct FeatureResults {
	FeaturePoints keypoints;
	cv::Mat descriptors;
};

enum FeatureDetectionType {
	ORB,
	SIFT,
};

template<size_t N>
std::unique_ptr<PointContainer<int, N>> convertToCoordinatePos(cv::Mat& mat, PointContainer<float, N>& src);

std::unique_ptr<FeatureResults> findFeatures(const cv::Mat& img, const FeatureDetectionType& featureType);
std::unique_ptr<MatchesContainer> findMatches(const FeatureDetectionType& featureType, const cv::Mat& descriptors1, const cv::Mat& descriptors2);
std::unique_ptr<MatchesContainer> findMatchesKNN(const FeatureDetectionType& featureType, const cv::Mat& descriptors1, const cv::Mat& descriptors2);

std::unique_ptr<MatchesContainer> filterToGoodMatchesKD(const MatchesContainerKNN& matches);

cv::Mat getHomography(const cv::Mat& img1, const cv::Mat& img2, const FeatureDetectionType& featureType = ORB);
std::unique_ptr<CornerPointsContainerFloat> getTargetObjectCorners(const cv::Mat& img);
std::unique_ptr<CornerPointsContainerFloat> objectCornerPointsToSceneCornerPoints(const cv::Mat& homography, const CornerPointsContainerFloat& objectCornerPoints);

template<typename T, size_t N = 4>
void markCornersAndOutlineObject(cv::Mat& dst, const PointContainer<T, N>& cornerPoints);

void printUsage() {
	std::cout << "Usage: " << std::endl;
	std::cout << " ObjectDetector <object image> <scene image> <method>" << std::endl;
	std::cout << " <object image> an image of the object to be detected" << std::endl;
	std::cout << " <scene image> an image of a scene to search for the object" << std::endl;
	std::cout << " <method>  SIFT or ORB detection" << std::endl;
	std::cout << " e.g.: ObjectDetector object.png scene.png SIFT" << std::endl;
}

std::string toLower(const std::string& str) {
	std::string result = str;
	std::transform(result.begin(), result.end(), result.begin(),
		[](unsigned char c) { return std::tolower(c); });

	return result;
}

int main(int argc, char* argv[]) {
	Timer timer;

	if (argc != 4) {
		printUsage();
		exit(-1);
	}

	cv::Mat objImage = cv::imread(argv[1]);
	if (objImage.empty()) {
		std::cerr << "Failed to read image from " << argv[1] << std::endl;
		exit(-2);
	}

	cv::Mat scnImage = cv::imread(argv[2]);
	if (scnImage.empty()) {
		std::cerr << "Failed to read image from " << argv[2] << std::endl;
		exit(-3);
	}

	std::string method = toLower(argv[3]);

	if (method != "sift" && method != "orb") {
		std::cerr << "Invalid method '" << argv[3] << "'" << std::endl;
		exit(-4);
	}
	cv::Mat detImage = scnImage.clone();
	cv::Mat d1Img = scnImage.clone();

	FeatureDetectionType detectionType = ORB;
	if (method == "sift")
		detectionType = SIFT;
	else if (method == "orb")
		detectionType = ORB;

	const auto homography = getHomography(objImage, scnImage, detectionType);

	if (verbose)
		std::cout << homography << std::endl;

	const auto objectCorners = getTargetObjectCorners(objImage);

	if (verbose) {
		for (const auto& cornerPoints : *objectCorners)
			std::cout << cornerPoints << std::endl;
	}

	const auto sceneCorners = objectCornerPointsToSceneCornerPoints(homography, *objectCorners);

	if (verbose) {
		for (const auto& cornerPoints : *sceneCorners)
			std::cout << cornerPoints << std::endl;
	}

	markCornersAndOutlineObject(detImage, *sceneCorners);

	// Save the detected object
	cv::imwrite("detectedObject.png", detImage);
	cv::namedWindow("Detection");
	cv::imshow("Detection", detImage);
	//std::cout << "That took " << timer.elapsed() << " seconds" << std::endl;
	cv::waitKey();

}

inline cv::Ptr<cv::FeatureDetector> getDetector(const FeatureDetectionType& featureType) {
	if (featureType == ORB)
		return cv::ORB::create(orbFeatures);
	if (featureType == SIFT)
		return cv::SIFT::create();

	return cv::ORB::create();
}

inline cv::Ptr<cv::DescriptorMatcher> getMatcher(const FeatureDetectionType& featureType) {
	if (featureType == ORB)
		return cv::BFMatcher::create(cv::NORM_HAMMING);
	if (featureType == SIFT)
		return cv::BFMatcher::create();

	return cv::BFMatcher::create();
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

template<size_t N>
std::unique_ptr<PointContainer<int, N>> convertToCoordinatePos(cv::Mat& mat, PointContainer<float, N>& src) {
	auto result = std::make_unique<PointContainer<int, N>>();

	for (size_t i = 0; i < N; i++) {
		result.get()[0][i] = convertToCoordinate(src[i]);
	}

	return result;
}

// ASSIGNMENT STEP 1
std::unique_ptr<FeatureResults> findFeatures(const cv::Mat& img, const FeatureDetectionType& featureType) {
	const auto detector = getDetector(featureType);

	auto results = std::make_unique<FeatureResults>();

	const Timer timer;

	detector->detectAndCompute(img, cv::noArray(), results->keypoints, results->descriptors);

	std::cout << results->keypoints.size() << " features detected in " << timer.elapsed() << " seconds" << std::endl;

	return results;
}

/*
 * ASSIGNMENT STEP 2
 *
 * Used for ORB
 */
std::unique_ptr<MatchesContainer> findMatches(const FeatureDetectionType& featureType, const cv::Mat& descriptors1, const cv::Mat& descriptors2) {
	static auto matcher = getMatcher(featureType);

	auto result = std::make_unique<MatchesContainer>();

	const Timer timer;

	matcher->match(descriptors1, descriptors2, *result);

	std::cout << result->size() << " matches found in " << timer.elapsed() << " seconds" << std::endl;

	return result;
}

/*
 * ASSIGNMENT STEP 2 (with STEP 3)
 *
 * Used for SIFT
 */
std::unique_ptr<MatchesContainer> findMatchesKNN(const FeatureDetectionType& featureType, const cv::Mat& descriptors1, const cv::Mat& descriptors2) {
	static auto matcher = getMatcher(featureType);

	const auto result = std::make_unique<MatchesContainerKNN>();

	const Timer timer;

	matcher->knnMatch(descriptors1, descriptors2, *result, 2);

	std::cout << result->size() << " matches found in " << timer.elapsed() << " seconds" << std::endl;

	auto filtered_results = filterToGoodMatchesKD(*result);
	std::cout << "Filtered to " << filtered_results->size() << " good matches, now up to " << timer.elapsed() << " seconds" << std::endl;

	return filtered_results;
}

// ASSIGNMENT STEP 3
std::unique_ptr<MatchesContainer> filterToGoodMatchesKD(const MatchesContainerKNN& matches) {
	auto result = std::make_unique<MatchesContainer>();

	for (const auto& match : matches) {
		if (match[0].distance < 0.8 * match[1].distance) {
			result->push_back(match[0]);
		}
	}

	return result;
}


// ASSIGNMENT STEP 3
std::unique_ptr<TransformMatches> getGoodPoints(const MatchesContainer& matches, const FeaturePoints& keypoints1, const FeaturePoints& keypoints2) {
	auto results = std::make_unique<TransformMatches>();

	for (const auto& match : matches) {
		results->first.push_back(keypoints1[match.queryIdx].pt);
		results->second.push_back(keypoints2[match.trainIdx].pt);
	}

	return results;
}

// ASSIGNMENT STEP 4
cv::Mat getHomography(const cv::Mat& img1, const cv::Mat& img2, const FeatureDetectionType& featureType) {

	const auto img1_features = findFeatures(img1, featureType);
	const auto img2_features = findFeatures(img2, featureType);

	std::unique_ptr<MatchesContainer> matches_result;

	if (featureType == ORB || (featureType == SIFT && !kDTreeEnabled)) {
		auto matches = findMatches(featureType, img1_features->descriptors, img2_features->descriptors);

		matches_result = std::move(matches);
	}
	else if (featureType == SIFT) {
		auto matches = findMatchesKNN(featureType, img1_features->descriptors, img2_features->descriptors);

		matches_result = std::move(matches);
	}

	const auto goodPoints =
		getGoodPoints(*matches_result, img1_features->keypoints, img2_features->keypoints);

	std::vector<unsigned char> inliers;

	auto homography = cv::findHomography(goodPoints->first, goodPoints->second, inliers, cv::RANSAC);

	return homography;
}

/*
 * ASSIGNMENT STEP 5
 *
 * Corners of the target are just the corners of the object image
 * Array indexes:
 * - 0: Top-left
 * - 1: Top-right
 * - 2: Bottom-right
 * - 3: Bottom-left
 */
std::unique_ptr<CornerPointsContainerFloat> getTargetObjectCorners(const cv::Mat& img) {
	const auto img_bounds = std::make_pair(static_cast<float>(img.size().width - 1), static_cast<float>(img.size().height - 1));

	auto corners = std::make_unique<CornerPointsContainerFloat>(CornerPointsContainerFloat{{
		{0, 0},
		{img_bounds.first, 0},
		{img_bounds.first, img_bounds.second},
		{0, img_bounds.second},
	}});

	return corners;
}

/*
 * ASSIGNMENT STEP 6
 *
 * Translate object corner points to scene corner points
 */
std::unique_ptr<CornerPointsContainerFloat> objectCornerPointsToSceneCornerPoints(const cv::Mat& homography, const CornerPointsContainerFloat& objectCornerPoints) {
	auto corners = std::make_unique<CornerPointsContainerFloat>();

	cv::perspectiveTransform(objectCornerPoints, *corners, homography);

	return corners;
}

/*
 * ASSIGNMENT STEP 7
 *
 * - Draw a small circle at each detected object corner
 * - Draw lines between the object corners, to outline the detected object
 */
template<typename T, size_t N = 4>
void markCornersAndOutlineObject(cv::Mat& dst, const PointContainer<T, N>& cornerPoints) {
	static auto markColor = cv::Scalar(0, 255, 0);
	static int thickness = 5;

	for (int i = 0; i < N; ++i) {
		cv::line(dst, cornerPoints[i], cornerPoints[(i + 1) % cornerPoints.size()], markColor, thickness);

		cv::circle(dst, cornerPoints[i], thickness, markColor, thickness);
	}
}
