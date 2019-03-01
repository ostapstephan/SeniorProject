#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <boost/thread/thread.hpp>
#include <boost/chrono/chrono.hpp>
#include <csignal>
#include <iostream>
#include <thread>
#include <future>

#include "pupiltracker/PupilTracker.h"
#include "pupiltracker/cvx.h"
#include "findEyeCenter.h"

volatile unsigned long mainThread;

unsigned long getThreadId(){
	std::string threadId = boost::lexical_cast<std::string>(boost::this_thread::get_id());
	unsigned long threadNumber = 0;
	sscanf(threadId.c_str(), "%lx", &threadNumber);
	return threadNumber;
}

static void handler(int signo) {
	if (signo == 30) {
		if (getThreadId() != mainThread) {
			/* throw boost::thread_interrupted(); */
		}
		std::cout<<getThreadId()<<"\n";
	}
}

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

	void tryFindPupilEllipse(cv::Mat *eye, cv::RotatedRect *el) {
		cv::Mat1f mat = cv::Mat1f::zeros(1, 5);

		pupiltracker::tracker_log log;

		pupiltracker::TrackerParams pupil_tracker_params;
		pupil_tracker_params.Radius_Min = 20;
		pupil_tracker_params.Radius_Max = 80;
		pupil_tracker_params.CannyThreshold1 = 20;
		pupil_tracker_params.CannyThreshold2 = 40;
		pupil_tracker_params.CannyBlur = 1.6;
		pupil_tracker_params.EarlyRejection = true;
		pupil_tracker_params.EarlyTerminationPercentage = 95;
		pupil_tracker_params.PercentageInliers = 20;
		pupil_tracker_params.InlierIterations = 3;
		pupil_tracker_params.ImageAwareSupport = true;
		pupil_tracker_params.StarburstPoints = 0;
		//pupil_tracker_params.Seed = 0;

		pupiltracker::findPupilEllipse_out pupil_tracker_out;
		bool found = false;
		try {
			found = pupiltracker::findPupilEllipse(pupil_tracker_params, *eye, pupil_tracker_out, log);
		} catch (boost::thread_interrupted &e) {
			printf("INTERRUPTED\n");
		}
		if (found)
			*el = pupil_tracker_out.elPupil;

	}

	tuple findPupilEllipse(cv::Mat eye) {
		sigset_t sigset;
		if (signal(30, handler) == SIG_ERR) {
			fprintf(stderr, "Could not put handle on signal 30: %s\n",strerror(errno));
			return make_tuple(-1);
		}
		if (sigaddset(&sigset, 11) == -1) {
				fprintf(stderr, "Could not add signal 11 to sigset: %s\n",strerror(errno));
				return make_tuple(-1);
		}
		if (sigprocmask(SIG_BLOCK, &sigset, NULL) == -1) {
			fprintf(stderr, "Could not block signals 11 and 30: %s\n",strerror(errno));
			return make_tuple(-1);
		}
		mainThread = getThreadId();
		/* boost::this_thread::disable_interruption di; */

		cv::RotatedRect el = cv::RotatedRect(cv::Point2f(0,0), cv::Size2f(0,0), 0);
		boost::thread track = boost::thread(tryFindPupilEllipse, &eye, &el);
		pthread_t thd = track.native_handle();

		if (track.try_join_for(boost::chrono::milliseconds(100))) {
			printf("TRUE\n");
		} else {
			printf("FALSE\n");
			track.interrupt();
			/* track.detach(); */
			/* pthread_cancel(thd); */
			pthread_kill(thd, 30);
		}

		if (sigprocmask(SIG_UNBLOCK, &sigset, NULL) == -1) {
			fprintf(stderr, "Could not unblock signal 11 and 30: %s\n",strerror(errno));
			return make_tuple(-1);
		}
		return make_tuple(el.center.x, el.center.y, el.size.width/2, el.size.height/2, el.angle);

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
