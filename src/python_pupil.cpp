#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

#include "pupiltracker/PupilTracker.h"
#include "pupiltracker/cvx.h"
#include "findEyeCenter.h"

namespace pbcvt {

	using namespace boost::python;

	tuple findPupil(cv::Mat face, int ex, int ey, int ew, int eh) {

		cv::Rect eyeROI;
		eyeROI.x = ex;
		eyeROI.y = ey;
		eyeROI.width = ew;
		eyeROI.height = eh;
		
		cv::Point p = findEyeCenter(face, eyeROI, "");
		tuple ret = make_tuple(p.x,p.y);

		return ret;
	}

	tuple findPupilEllipse(cv::Mat eye) {
		cv::RotatedRect el;
		tuple ret;

		pupiltracker::tracker_log log;

		pupiltracker::TrackerParams pupil_tracker_params;
		pupil_tracker_params.Radius_Min = 20;
		pupil_tracker_params.Radius_Max = 70;
		pupil_tracker_params.CannyThreshold1 = 20;
		pupil_tracker_params.CannyThreshold2 = 40;
		pupil_tracker_params.CannyBlur = 1.6;
		pupil_tracker_params.EarlyRejection = true;
		pupil_tracker_params.EarlyTerminationPercentage = 95;
		pupil_tracker_params.PercentageInliers = 20;
		pupil_tracker_params.InlierIterations = 2;
		pupil_tracker_params.ImageAwareSupport = true;
		pupil_tracker_params.StarburstPoints = 0;
		//pupil_tracker_params.Seed = 0;

		pupiltracker::findPupilEllipse_out pupil_tracker_out;
		bool found = pupiltracker::findPupilEllipse(pupil_tracker_params, eye, pupil_tracker_out, log);

		if (found) {
			el = pupil_tracker_out.elPupil;
			/* el.center -= cv::Point2f(eye.cols, eye.rows)/2; */
			ret = make_tuple(el.center.x,el.center.y,el.size.width,el.size.height,el.angle);
		} else {
			ret = make_tuple(0);
		}

		return ret;

	}

#if (PY_VERSION_HEX >= 0x03000000)

	static void *init_ar() {
#else
		static void init_ar(){
#endif
		Py_Initialize();

		import_array();
		return NUMPY_IMPORT_ARRAY_RETVAL;
	}

	BOOST_PYTHON_MODULE (pbcvt) {
		//using namespace XM;
		init_ar();

		//initialize converters
		to_python_converter<cv::Mat,pbcvt::matToNDArrayBoostConverter>();
		matFromNDArrayBoostConverter();

		//expose module-level functions
		def("findPupil", findPupil);
		def("findPupilEllipse", findPupilEllipse);

	}

} //end namespace pbcvt
