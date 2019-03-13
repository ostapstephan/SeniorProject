#include <opencv2/ccalib/multicalib.hpp>

using namespace cv;

int main(int argc, char *argv[]) {
	if (argc != 4)
		std::cout << "usage: " << argv[0] << " [# cameras] [input_xml_file] [output_xml_file]\n";

	int cameraType = multicalib::MultiCameraCalibration::PINHOLE;
	int nCamera = atoi(argv[1]);
	std::string inputFilename = argv[2];
	std::string outputFilename = argv[3];
	float patternWidth = 1920;
	float patternHeight = 1080;
	TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.001);
	cv::multicalib::MultiCameraCalibration multiCalib(cameraType, nCamera, inputFilename, patternWidth, patternHeight, 0, 1, 1, 0, criteria);
	multiCalib.run();
	multiCalib.writeParameters(outputFilename);

	return 0;
}
