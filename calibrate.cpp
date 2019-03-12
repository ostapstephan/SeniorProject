#include <opencv2/ccalib/multicalib.hpp>

using namespace cv;

int main() {

	int cameraType = multicalib::MultiCameraCalibration::PINHOLE;
	int nCamera = 4;
	std::string inputFilename = "calibrate/*";
	imagelist_creator
	std::string outputFilename = "calibratedcams.xml";
	float patternWidth = 0;
	float patternHeight = 0;
	cv::multicalib::MultiCameraCalibration multiCalib(cameraType, nCamera, inputFilename, patternWidth, patternHeight);
	multiCalib.run();
	multiCalib.writeParameters(outputFilename);

	return 0;
}
