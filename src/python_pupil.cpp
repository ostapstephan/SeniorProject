#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

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

	}

} //end namespace pbcvt
