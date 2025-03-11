#include "Timer.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <string>

struct FeatureResults {
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
};

enum FeatureDetectionType {
	ORB,
	SIFT,
};

std::unique_ptr<cv::Mat> getHomography(const cv::Mat& img1, const cv::Mat& img2, const FeatureDetectionType& featureType = ORB);

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

	FeatureDetectionType detectionType = ORB;
	if (method == "sift")
		detectionType = SIFT;
	else if (method == "orb")
		detectionType = ORB;

	auto homography = getHomography(objImage, scnImage, detectionType);

	// Save the detected object
	cv::imwrite("detectedObject.png", detImage);
	cv::namedWindow("Detection");
	cv::imshow("Detection", detImage);
	std::cout << "That took " << timer.elapsed() << " seconds" << std::endl;
	cv::waitKey();
		 
}

inline cv::Ptr<cv::FeatureDetector> getDetector(const FeatureDetectionType& featureType) {
	if (featureType == ORB)
		return cv::ORB::create();
	if (featureType == SIFT)
		return cv::SIFT::create();

	return cv::ORB::create();
}

inline cv::Ptr<cv::DescriptorMatcher> getMatcher() {
	return cv::BFMatcher::create();
}

std::unique_ptr<FeatureResults> findFeatures(const cv::Mat& img, const FeatureDetectionType& featureType) {
	auto detector = getDetector(featureType);

	auto results = std::make_unique<FeatureResults>();

	detector->detectAndCompute(img, cv::noArray(), results->keypoints, results->descriptors);

	return results;
}

template<typename MatchesContainer = std::vector<std::vector<cv::DMatch>>>
std::unique_ptr<MatchesContainer> findMatches(const cv::Mat& descriptors1, const cv::Mat& descriptors2) {
	static auto matcher = getMatcher();

	auto result = std::make_unique<MatchesContainer>();

	matcher->knnMatch(descriptors1, descriptors2, *result, 2);

	return result;
}

template<typename GoodMatchesContainer = std::vector<cv::DMatch>,
		 typename MatchesContainer     = std::vector<std::vector<cv::DMatch>>>
std::unique_ptr<GoodMatchesContainer> findGoodMatches(const MatchesContainer& matches) {
	auto results = std::make_unique<GoodMatchesContainer>();
	for (const auto& match : matches) {
		if (match[0].distance < 0.8 * match[1].distance)
			results->push_back(match[0]);
	}

	return results;
}

template<typename TransformPoints      = std::vector<cv::Point2f>,
		 typename TransformMatches     = std::pair<TransformPoints, TransformPoints>,
		 typename GoodMatchesContainer = std::vector<cv::DMatch>,
		 typename FeatureLocations     = std::vector<cv::KeyPoint>>
std::unique_ptr<TransformMatches> getGoodPoints(const GoodMatchesContainer& matches, const FeatureLocations& keypoints1, const FeatureLocations& keypoints2) {
	auto results = std::make_unique<TransformMatches>();

	for (const auto& match : matches) {
		results->first.push_back(keypoints1[match.queryIdx].pt);
		results->second.push_back(keypoints2[match.trainIdx].pt);
	}

	return results;
}

std::unique_ptr<cv::Mat> getHomography(const cv::Mat& img1, const cv::Mat& img2, const FeatureDetectionType& featureType) {

	const auto img1_features = findFeatures(img1, featureType);
	const auto img2_features = findFeatures(img2, featureType);

	const auto matches = findMatches(img1_features->descriptors, img2_features->descriptors);

	const auto goodMatches = findGoodMatches(*matches);

	const auto goodPoints =
		getGoodPoints(*goodMatches, img1_features->keypoints, img2_features->keypoints);

	std::vector<unsigned char> inliers;

	auto homography =
		std::make_unique<cv::Mat>(cv::findHomography(goodPoints->second, goodPoints->first, inliers, cv::RANSAC));

	return homography;
}

