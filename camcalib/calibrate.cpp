#include <opencv2/ccalib/multicalib.hpp>

using namespace cv;

int main(int argc, char *argv[]) {
	if (argc < 4) {
		std::cout << "usage: " << argv[0] << " [# cameras] [input_xml_file] [output_xml_file] [debug]\n";
        return 0;
    }

	int cameraType = multicalib::MultiCameraCalibration::PINHOLE;
	int nCamera = atoi(argv[1]);
	std::string inputFilename = argv[2];
	std::string outputFilename = argv[3];
	float patternWidth = 1920;
	float patternHeight = 1080;
	TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 5, 0.01);
	cv::multicalib::MultiCameraCalibration multiCalib(cameraType, nCamera, inputFilename, patternWidth, patternHeight, 1, argc==5, 1, 0, criteria);
	multiCalib.run();
	multiCalib.writeParameters(outputFilename);

	return 0;
}
