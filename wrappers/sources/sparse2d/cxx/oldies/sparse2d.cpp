#include "starlet2D.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(sparse2d_Sn)
{
	np::initialize();

	class_< Starlet2D >("Starlet2D", init<int, int, int,int,int,int >())
		.def("forward2d", &Starlet2D::forward2d)
		.def("forward2d_omp", &Starlet2D::forward2d_omp)
	  .def("backward2d_omp", &Starlet2D::backward2d_omp)
		.def("backward2d_omp_v2", &Starlet2D::backward2d_omp_v2)
		.def("forward2d_mr_ref_omp", &Starlet2D::forward2d_mr_ref_omp)
	  .def("filter", &Starlet2D::filter2d);
}
