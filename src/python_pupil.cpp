#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
/* #include <boost/thread/thread.hpp> */
/* #include <boost/chrono/chrono.hpp> */
#include <thread>
#include <chrono>
#include <sys/mman.h>
#include <csignal>
#include <sys/wait.h>

#include "pupiltracker/PupilTracker.h"
#include "pupiltracker/cvx.h"
#include "findEyeCenter.h"

namespace pbcvt {

	using namespace boost::python;


	/// @brief Type that allows for registration of conversions from
	///        python iterable types.
	struct iterable_converter
	{
	  /// @note Registers converter from a python interable type to the
	  ///       provided type.
	  template <typename Container>
	  iterable_converter&
	  from_python()
	  {
		boost::python::converter::registry::push_back(
		  &iterable_converter::convertible,
		  &iterable_converter::construct<Container>,
		  boost::python::type_id<Container>());

		// Support chaining.
		return *this;
	  }

	  /// @brief Check if PyObject is iterable.
	  static void* convertible(PyObject* object)
	  {
		return PyObject_GetIter(object) ? object : NULL;
	  }

	  /// @brief Convert iterable PyObject to C++ container type.
	  ///
	  /// Container Concept requirements:
	  ///
	  ///   * Container::value_type is CopyConstructable.
	  ///   * Container can be constructed and populated with two iterators.
	  ///     I.e. Container(begin, end)
	  template <typename Container>
	  static void construct(
		PyObject* object,
		boost::python::converter::rvalue_from_python_stage1_data* data)
	  {
		namespace python = boost::python;
		// Object is a borrowed reference, so create a handle indicting it is
		// borrowed for proper reference counting.
		python::handle<> handle(python::borrowed(object));

		// Obtain a handle to the memory block that the converter has allocated
		// for the C++ type.
		typedef python::converter::rvalue_from_python_storage<Container>
																	storage_type;
		void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

		typedef python::stl_input_iterator<typename Container::value_type>
																		iterator;

		// Allocate the C++ type into the converter's memory block, and assign
		// its handle to the converter's convertible variable.  The C++
		// container is populated by passing the begin and end iterators of
		// the python object to the container's constructor.
		new (storage) Container(
		  iterator(python::object(handle)), // begin
		  iterator());                      // end
		data->convertible = storage;
	  }
	};


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


	void tryFindPupilEllipse(cv::Mat &eye, cv::RotatedRect *el, pupiltracker::TrackerParams pupil_tracker_params) {
		pupiltracker::tracker_log log;
		pupiltracker::findPupilEllipse_out pupil_tracker_out;
		bool found = found = pupiltracker::findPupilEllipse(pupil_tracker_params, eye, pupil_tracker_out, log);
		if (found)
			*el = pupil_tracker_out.elPupil;
	}


	static cv::RotatedRect *ell;

	static void handler(int signo) {
		if (signo == 30){
			exit(0);
		}
	}

	static void timeout(pid_t p, int milli) {
		std::this_thread::sleep_for(std::chrono::milliseconds(milli));
		if (waitpid(-1, NULL, WNOHANG) == 0)
			kill(p, 30);
	}


	tuple findPupilEllipse(cv::Mat eye, int timeoutmilli,
								int Radius_Min,
								int Radius_Max,
								int CannyThreshold1,
								int CannyThreshold2,
								double CannyBlur,
								bool EarlyRejection,
								int EarlyTerminationPercentage,
								int PercentageInliers,
								int InlierIterations,
								bool ImageAwareSupport,
								int StarburstPoints
								/* ,int Seed */
							) {

		pupiltracker::TrackerParams pupil_tracker_params;
		pupil_tracker_params.Radius_Min = Radius_Min;
		pupil_tracker_params.Radius_Max = Radius_Max;
		pupil_tracker_params.CannyThreshold1 = CannyThreshold1;
		pupil_tracker_params.CannyThreshold2 = CannyThreshold2;
		pupil_tracker_params.CannyBlur = CannyBlur;
		pupil_tracker_params.EarlyRejection = EarlyRejection;
		pupil_tracker_params.EarlyTerminationPercentage = EarlyTerminationPercentage;
		pupil_tracker_params.PercentageInliers = PercentageInliers;
		pupil_tracker_params.InlierIterations = InlierIterations;
		pupil_tracker_params.ImageAwareSupport = ImageAwareSupport;
		pupil_tracker_params.StarburstPoints = StarburstPoints;
		//pupil_tracker_params.Seed = 0;
		ell = (cv::RotatedRect *) mmap(NULL, sizeof *ell, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
		/* cv::RotatedRect el = cv::RotatedRect(cv::Point2f(0,0), cv::Size2f(0,0), 0); */
		/* boost::thread track = boost::thread(tryFindPupilEllipse, eye, el); */
		/* pthread_t thd = track.native_handle(); */
		/* if (!track.try_join_for(boost::chrono::milliseconds(timeout))) */
		/* 	track.detach(); */
		pid_t proc = -1;
		sigset_t sigset;
		if (signal(30, handler) == SIG_ERR) {
			fprintf(stderr, "Could not put handle on signal 30: %s\n",strerror(errno));
			goto END;
		}
		if (sigemptyset(&sigset) == -1) {
			fprintf(stderr, "Could not init sigset: %s\n",strerror(errno));
			goto END;
		}
		if (sigaddset(&sigset, 30) == -1) {
			fprintf(stderr, "Could not add signal 30 to sigset: %s\n",strerror(errno));
			goto END;
		}
		if (sigprocmask(SIG_BLOCK, &sigset, NULL) == -1) {
			fprintf(stderr, "Could not block signal 30: %s\n",strerror(errno));
			goto END;
		}

		/* proc = fork(); */
		switch(proc) {
			case -1:
				/* fprintf(stderr, "Could not fork: %s\n", strerror(errno)); */
				tryFindPupilEllipse(eye, ell, pupil_tracker_params);
				break;
			case 0:
				tryFindPupilEllipse(eye, ell, pupil_tracker_params);
				exit(0);
				break;
			default:
				std::thread t = std::thread(timeout, proc, timeoutmilli);
				t.detach();
				wait(NULL);
				if (errno > 0 && errno != EINTR) {
					fprintf(stderr, "Problem waiting: %s\n", strerror(errno));
					/* return make_tuple(0,0,0,0,0); */
				}
				break;
		}

		if (sigprocmask(SIG_UNBLOCK, &sigset, NULL) == -1) {
			fprintf(stderr, "Could not unblock signal 30: %s\n",strerror(errno));
			/* return make_tuple(0,0,0,0,0); */
		}

	END:
		tuple ret = make_tuple(ell->center.x, ell->center.y, ell->size.width/2, ell->size.height/2, ell->angle);
		munmap(ell, sizeof *ell);
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
		// Register interable conversions.
		iterable_converter()
		// Build-in type.
		.from_python<std::vector<double>>()
		// Each dimension needs to be convertable.
		.from_python<std::vector<std::string>>()
		.from_python<std::vector<std::vector<std::string>>>();

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
