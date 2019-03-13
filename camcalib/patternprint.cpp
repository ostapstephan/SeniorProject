#include <opencv2/ccalib/randpattern.hpp>

using namespace cv;

int main() {

	int width = 1920;
	int height = 1080;
	cv::randpattern::RandomPatternGenerator generator(width, height);
	generator.generatePattern();
	Mat pattern = generator.getPattern();
	imshow("test", pattern);
	waitKey();
	imwrite("pattern.png", pattern);

	return 0;
}
